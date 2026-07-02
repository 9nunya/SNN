#include "brain.h"
#include "collections.h"
#include "compute/compute_ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace snn {

// ---- CUDA kernels ----

__global__ void synapse_step_kernel(
    const int* __restrict__ d_pre,
    const int* __restrict__ d_post,
    float* __restrict__ d_weight,
    float* __restrict__ d_g,
    float* __restrict__ d_eligibility,
    float* __restrict__ d_tau_s,
    const bool* __restrict__ d_old_spikes,
    float* __restrict__ d_I_syn,
    int S, float dt
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= S) return;

    int pre = d_pre[sid];
    bool fired = d_old_spikes ? d_old_spikes[pre] : false;
    float w = d_weight[sid];
    float I = fired ? w : 0.0f;

    float g = d_g[sid];
    float tau_s = d_tau_s[sid];
    float new_g = I + expf(-dt / tau_s) * (g - I);
    d_g[sid] = new_g;

    int post = d_post[sid];
    atomicAdd(d_I_syn + post, new_g);
}

__global__ void neuron_step_kernel(
    float* __restrict__ d_v,
    float* __restrict__ d_v_th,
    float* __restrict__ d_tau_rc,
    float* __restrict__ d_tau_ref,
    float* __restrict__ d_rest_time,
    bool* __restrict__ d_slf,
    int* __restrict__ d_type,
    float* __restrict__ d_w,
    float* __restrict__ d_tau_w,
    float* __restrict__ d_I_syn,
    const float* __restrict__ d_external_inputs,
    bool* __restrict__ d_new_spikes,
    int N, float dt
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= N) return;

    float I_ext = d_external_inputs ? d_external_inputs[nid] : 0.0f;
    float I_syn = d_I_syn ? d_I_syn[nid] : 0.0f;
    float I_total = I_ext + I_syn;

    // Load neuron state
    neuron_state<float> n;
    n.v = d_v[nid];
    n.v_th = d_v_th[nid];
    n.tau_rc = d_tau_rc[nid];
    n.tau_ref = d_tau_ref[nid];
    n.rest_time = d_rest_time[nid];
    n.slf = d_slf ? d_slf[nid] : false;
    n.type = d_type ? static_cast<neuron_type>(d_type[nid]) : neuron_type::LIF;

    // Adaptive state
    bool is_alif = (n.type == neuron_type::ALIF);
    adaptive_neuron_state<float> alif;
    if (is_alif && d_w) {
        alif.w = d_w[nid];
        alif.tau_w = d_tau_w[nid];
        alif.b = d_b ? d_b[nid] : 0.0f;
    }

    // Step neuron
    bool spiked = false;
    float effective_dt = dt - n.rest_time;
    if (effective_dt < 0.0f) effective_dt = 0.0f;
    if (effective_dt > dt) effective_dt = dt;

    if (n.rest_time > 0.0f) {
        n.rest_time -= dt;
    } else {
        n.v = I_total + (n.v - I_total) * expf(-effective_dt / n.tau_rc);
    }

    if (is_alif) {
        float effective_threshold = n.v_th + alif.w;
        if (n.v >= effective_threshold) {
            spiked = true;
            alif.w += alif.b;
            float st = effective_dt + n.tau_rc * logf((n.v - I_total) / (n.v_th - I_total));
            n.rest_time = n.tau_ref + st;
            n.v = n.slf ? n.v - n.v_th : 0.0f;
        }
        alif.w = alif.w * expf(-dt / alif.tau_w) + alif.b * spiked;
    } else {
        if (n.v >= n.v_th) {
            spiked = true;
            float st = effective_dt + n.tau_rc * logf((n.v - I_total) / (n.v_th - I_total));
            n.rest_time = n.tau_ref + st;
            n.v = n.slf ? n.v - n.v_th : 0.0f;
        }
    }

    d_v[nid] = n.v;
    d_rest_time[nid] = n.rest_time;
    d_new_spikes[nid] = spiked;
    if (is_alif && d_w) {
        d_w[nid] = alif.w;
        d_tau_w[nid] = alif.tau_w;
    }
}

__global__ void stdp_kernel(
    const int* __restrict__ d_pre,
    const int* __restrict__ d_post,
    float* __restrict__ d_weight,
    float* __restrict__ d_g,
    float* __restrict__ d_eligibility,
    float* __restrict__ d_tau_s,
    const bool* __restrict__ d_old_spikes,
    const bool* __restrict__ d_new_spikes,
    int S, float reward,
    float A_plus, float A_minus, float decay, float eta
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= S) return;

    int pre = d_pre[sid];
    int post = d_post[sid];
    bool pre_old = d_old_spikes ? d_old_spikes[pre] : false;
    bool post_new = d_new_spikes ? d_new_spikes[post] : false;

    float et = d_eligibility[sid];
    if (pre_old && !post_new) et += A_plus;
    else if (!pre_old && post_new) et += A_minus;
    et *= decay;
    d_eligibility[sid] = et;

    if (reward != 0.0f) {
        d_weight[sid] += reward * et * eta;
    }
}

// ---- single-brain brain_step_cuda ----

template<typename T>
void brain_step_cuda(brain<T>* b, const T* external_inputs, int num_inputs, T reward, T dt, int N, int S) {
    if (!b || !b->neuron_field) return;

    auto& g = b->gpu;
    if (!g.valid || g.N != N || g.S != S) {
        brain_gpu_cache_destroy(g);
        brain_gpu_cache_build(g, b);
    }

    if (!g.valid) return;

    // Copy old spikes host->device
    if (b->last_spikes && g.d_old_spikes && N > 0) {
        cudaMemcpy(g.d_old_spikes, b->last_spikes, N * sizeof(bool), cudaMemcpyHostToDevice);
    }

    float* d_external = nullptr;
    if (external_inputs && num_inputs > 0 && N > 0) {
        cudaMalloc(&d_external, N * sizeof(float));
        cudaMemcpy(d_external, external_inputs, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Phase 1: synapse step (uses d_old_spikes)
    if (g.S > 0) {
        if (g.d_I_syn) cudaMemset(g.d_I_syn, 0, N * sizeof(float));
        int threads = 256;
        int blocks = (g.S + threads - 1) / threads;
        synapse_step_kernel<<<blocks, threads>>>(
            g.d_pre, g.d_post,
            b->synapses->device_data.weight,
            b->synapses->device_data.g,
            b->synapses->device_data.eligibility,
            b->synapses->device_data.tau_s,
            g.d_old_spikes,
            g.d_I_syn,
            g.S, dt
        );
        cudaDeviceSynchronize();
    }

    // Phase 2: neuron step
    if (g.N > 0) {
        int threads = 256;
        int blocks = (g.N + threads - 1) / threads;
        neuron_step_kernel<<<blocks, threads>>>(
            b->neurons->device_data.v,
            b->neurons->device_data.v_th,
            b->neurons->device_data.tau_rc,
            b->neurons->device_data.tau_ref,
            b->neurons->device_data.rest_time,
            b->neurons->device_data.slf,
            b->neurons->device_data.type,
            b->neurons->device_data.w,
            b->neurons->device_data.tau_w,
            g.d_I_syn,
            d_external,
            g.d_new_spikes,
            g.N, dt
        );
        cudaDeviceSynchronize();
    }

    // Phase 3: R-STDP (synapse-based, O(S))
    if (g.S > 0) {
        int threads = 256;
        int blocks = (g.S + threads - 1) / threads;
        stdp_kernel<<<blocks, threads>>>(
            g.d_pre, g.d_post,
            b->synapses->device_data.weight,
            b->synapses->device_data.g,
            b->synapses->device_data.eligibility,
            b->synapses->device_data.tau_s,
            g.d_old_spikes,
            g.d_new_spikes,
            g.S, reward,
            0.1f, -0.1f, 0.95f, 0.01f
        );
        cudaDeviceSynchronize();
    }

    // Copy back new spikes device->host
    if (b->last_spikes && g.d_new_spikes && N > 0) {
        cudaMemcpy(b->last_spikes, g.d_new_spikes, N * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // Copy back weight/g/eligibility to host-side brain_synapse objects
    if (g.S > 0) {
        float* h_weight = new float[g.S];
        float* h_g = new float[g.S];
        float* h_elig = new float[g.S];
        cudaMemcpy(h_weight, b->synapses->device_data.weight, g.S * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_g, b->synapses->device_data.g, g.S * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_elig, b->synapses->device_data.eligibility, g.S * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < g.S; ++i) {
            b->neuron_field->synapses[i]->weight = h_weight[i];
            b->neuron_field->synapses[i]->s->s->weight = h_weight[i];
            b->neuron_field->synapses[i]->eligibility_trace = h_elig[i];
        }
        delete[] h_weight;
        delete[] h_g;
        delete[] h_elig;
    }

    if (d_external) cudaFree(d_external);
}

// ---- batched brain_step_batch ----

template<typename T>
void brain_step_batch(brain<T>* const* brains, int num_brains,
                      const T* const* external_inputs, int num_inputs,
                      T reward, T dt) {
    if (!brains || num_brains == 0) return;

    // Filter valid brains and build idx map
    std::vector<int> idx_map(num_brains, -1);
    std::vector<int> valid_N, valid_S;
    int total_N = 0, total_S = 0;
    for (int i = 0; i < num_brains; ++i) {
        if (!brains[i] || !brains[i]->neuron_field) continue;
        int Ni = brains[i]->neuron_field->num_neurons;
        int Si = brains[i]->neuron_field->num_synapses;
        if (Ni == 0) continue;
        idx_map[i] = (int)valid_N.size();
        valid_N.push_back(Ni);
        valid_S.push_back(Si);
        total_N += Ni;
        total_S += Si;
    }

    int valid_count = (int)valid_N.size();
    if (valid_count == 0) return;

    // Build offsets
    std::vector<int> neuron_offsets(valid_count + 1), syn_offsets(valid_count + 1);
    neuron_offsets[0] = 0;
    syn_offsets[0] = 0;
    for (int i = 0; i < valid_count; ++i) {
        neuron_offsets[i + 1] = neuron_offsets[i] + valid_N[i];
        syn_offsets[i + 1] = syn_offsets[i] + valid_S[i];
    }

    // Allocate flat device arrays
    int *d_neuron_offsets = nullptr, *d_syn_offsets = nullptr;
    float *d_v = nullptr, *d_v_th = nullptr, *d_tau_rc = nullptr, *d_tau_ref = nullptr, *d_rest_time = nullptr;
    bool *d_slf = nullptr; int *d_type = nullptr;
    float *d_w = nullptr, *d_tau_w = nullptr;
    float *d_old_spikes = nullptr, *d_new_spikes = nullptr, *d_I_syn = nullptr;
    float *d_external = nullptr;
    float *d_g = nullptr, *d_eligibility = nullptr, *d_weight = nullptr, *d_tau_s = nullptr;
    int *d_pre = nullptr, *d_post = nullptr;

    cudaMalloc(&d_neuron_offsets, (valid_count + 1) * sizeof(int));
    cudaMalloc(&d_syn_offsets, (valid_count + 1) * sizeof(int));

    if (total_N > 0) {
        cudaMalloc(&d_v, total_N * sizeof(float));
        cudaMalloc(&d_v_th, total_N * sizeof(float));
        cudaMalloc(&d_tau_rc, total_N * sizeof(float));
        cudaMalloc(&d_tau_ref, total_N * sizeof(float));
        cudaMalloc(&d_rest_time, total_N * sizeof(float));
        cudaMalloc(&d_slf, total_N * sizeof(bool));
        cudaMalloc(&d_type, total_N * sizeof(int));
        cudaMalloc(&d_w, total_N * sizeof(float));
        cudaMalloc(&d_tau_w, total_N * sizeof(float));
        cudaMalloc(&d_old_spikes, total_N * sizeof(bool));
        cudaMalloc(&d_new_spikes, total_N * sizeof(bool));
        cudaMalloc(&d_I_syn, total_N * sizeof(float));
        if (num_inputs > 0) cudaMalloc(&d_external, total_N * sizeof(float));
    }

    if (total_S > 0) {
        cudaMalloc(&d_pre, total_S * sizeof(int));
        cudaMalloc(&d_post, total_S * sizeof(int));
        cudaMalloc(&d_weight, total_S * sizeof(float));
        cudaMalloc(&d_g, total_S * sizeof(float));
        cudaMalloc(&d_eligibility, total_S * sizeof(float));
        cudaMalloc(&d_tau_s, total_S * sizeof(float));
    }

    // Pack host data into flat arrays
    std::vector<float> h_v(total_N), h_v_th(total_N), h_tau_rc(total_N), h_tau_ref(total_N), h_rest(total_N);
    std::vector<bool> h_slf(total_N);
    std::vector<int> h_type(total_N);
    std::vector<float> h_w(total_N), h_tau_w(total_N);
    std::vector<bool> h_old_spikes(total_N);
    std::vector<int> h_pre(total_S), h_post(total_S);
    std::vector<float> h_weight(total_S), h_g(total_S), h_elig(total_S), h_tau_s(total_S);
    std::vector<float> h_external(total_N, 0.0f);

    for (int i = 0; i < num_brains; ++i) {
        if (idx_map[i] < 0) continue;
        int bi = idx_map[i];
        int noff = neuron_offsets[bi];
        int soff = syn_offsets[bi];
        int Ni = valid_N[bi];
        int Si = valid_S[bi];
        auto* b = brains[i];

        // Pack synaptic structure
        for (int s = 0; s < Si; ++s) {
            int idx = soff + s;
            auto* bs = b->neuron_field->synapses[s];
            h_pre[idx] = bs->pre_neuron_idx + noff;
            h_post[idx] = bs->post_neuron_idx + noff;
            h_weight[idx] = bs->weight;
            h_g[idx] = bs->s->s->g;
            h_elig[idx] = bs->eligibility_trace;
            h_tau_s[idx] = bs->s->s->tau_s;
        }

        // Pack neuron structure from device
        if (Ni > 0) {
            auto& nd = b->neurons->device_data;
            cudaMemcpy(h_v.data() + noff, nd.v, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v_th.data() + noff, nd.v_th, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tau_rc.data() + noff, nd.tau_rc, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tau_ref.data() + noff, nd.tau_ref, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_rest.data() + noff, nd.rest_time, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_slf.data() + noff, nd.slf, Ni * sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_type.data() + noff, nd.type, Ni * sizeof(int), cudaMemcpyDeviceToHost);
            if (nd.w) cudaMemcpy(h_w.data() + noff, nd.w, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            if (nd.tau_w) cudaMemcpy(h_tau_w.data() + noff, nd.tau_w, Ni * sizeof(float), cudaMemcpyDeviceToHost);
            if (b->last_spikes) {
                for (int n = 0; n < Ni; ++n) h_old_spikes[noff + n] = b->last_spikes[n];
            }
        }
    }

    // Copy packed data to device
    cudaMemcpy(d_neuron_offsets, neuron_offsets.data(), (valid_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syn_offsets, syn_offsets.data(), (valid_count + 1) * sizeof(int), cudaMemcpyHostToDevice);

    if (total_N > 0) {
        cudaMemcpy(d_v, h_v.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v_th, h_v_th.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tau_rc, h_tau_rc.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tau_ref, h_tau_ref.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rest_time, h_rest.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slf, h_slf.data(), total_N * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_type, h_type.data(), total_N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tau_w, h_tau_w.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_old_spikes, h_old_spikes.data(), total_N * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemset(d_new_spikes, 0, total_N * sizeof(bool));
        cudaMemset(d_I_syn, 0, total_N * sizeof(float));
        if (num_inputs > 0 && d_external) {
            cudaMemcpy(d_external, h_external.data(), total_N * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    if (total_S > 0) {
        cudaMemcpy(d_pre, h_pre.data(), total_S * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_post, h_post.data(), total_S * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, h_weight.data(), total_S * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g.data(), total_S * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_eligibility, h_elig.data(), total_S * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tau_s, h_tau_s.data(), total_S * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Phase 1: synapse step
    if (total_S > 0) {
        int threads = 256;
        int blocks = (total_S + threads - 1) / threads;
        synapse_step_kernel<<<blocks, threads>>>(
            d_pre, d_post, d_weight, d_g, d_eligibility, d_tau_s,
            d_old_spikes, d_I_syn, total_S, dt
        );
        cudaDeviceSynchronize();
    }

    // Phase 2: neuron step
    if (total_N > 0) {
        int threads = 256;
        int blocks = (total_N + threads - 1) / threads;
        neuron_step_kernel<<<blocks, threads>>>(
            d_v, d_v_th, d_rest_time, d_slf, d_type,
            d_w, d_tau_w,
            d_I_syn, d_external, d_new_spikes,
            total_N, dt
        );
        cudaDeviceSynchronize();
    }

    // Phase 3: R-STDP (synapse-based O(S))
    if (total_S > 0) {
        int threads = 256;
        int blocks = (total_S + threads - 1) / threads;
        stdp_kernel<<<blocks, threads>>>(
            d_pre, d_post, d_weight, d_g, d_eligibility, d_tau_s,
            d_old_spikes, d_new_spikes, total_S, reward,
            0.1f, -0.1f, 0.95f, 0.01f
        );
        cudaDeviceSynchronize();
    }

    // Copy results back to hosts and scatter to brains
    if (total_N > 0) {
        cudaMemcpy(h_v.data(), d_v, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v_th.data(), d_v_th, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tau_rc.data(), d_tau_rc, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tau_ref.data(), d_tau_ref, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rest.data(), d_rest_time, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_w.data(), d_w, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tau_w.data(), d_tau_w, total_N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_old_spikes.data(), d_new_spikes, total_N * sizeof(bool), cudaMemcpyDeviceToHost);
    }
    if (total_S > 0) {
        cudaMemcpy(h_weight.data(), d_weight, total_S * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_g.data(), d_g, total_S * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_elig.data(), d_eligibility, total_S * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Scatter back to individual brains
    for (int i = 0; i < num_brains; ++i) {
        if (idx_map[i] < 0) continue;
        int bi = idx_map[i];
        int noff = neuron_offsets[bi];
        int soff = syn_offsets[bi];
        int Ni = valid_N[bi];
        int Si = valid_S[bi];
        auto* b = brains[i];

        // Neuron device state
        if (Ni > 0) {
            auto& nd = b->neurons->device_data;
            cudaMemcpy(nd.v, h_v.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(nd.v_th, h_v_th.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(nd.tau_rc, h_tau_rc.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(nd.tau_ref, h_tau_ref.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(nd.rest_time, h_rest.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            if (nd.w) cudaMemcpy(nd.w, h_w.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            if (nd.tau_w) cudaMemcpy(nd.tau_w, h_tau_w.data() + noff, Ni * sizeof(float), cudaMemcpyHostToDevice);
            // Spikes
            if (b->last_spikes) {
                for (int n = 0; n < Ni; ++n) {
                    b->last_spikes[n] = h_old_spikes[noff + n];
                }
            }
        }

        // Synapse host-side objects
        for (int s = 0; s < Si; ++s) {
            int idx = soff + s;
            b->neuron_field->synapses[s]->weight = h_weight[idx];
            b->neuron_field->synapses[s]->s->s->weight = h_weight[idx];
            b->neuron_field->synapses[s]->eligibility_trace = h_elig[idx];
        }
    }

    cleanup:
    cudaFree(d_neuron_offsets);
    cudaFree(d_syn_offsets);
    if (d_v) cudaFree(d_v);
    if (d_v_th) cudaFree(d_v_th);
    if (d_tau_rc) cudaFree(d_tau_rc);
    if (d_tau_ref) cudaFree(d_tau_ref);
    if (d_rest_time) cudaFree(d_rest_time);
    if (d_slf) cudaFree(d_slf);
    if (d_type) cudaFree(d_type);
    if (d_w) cudaFree(d_w);
    if (d_tau_w) cudaFree(d_tau_w);
    if (d_old_spikes) cudaFree(d_old_spikes);
    if (d_new_spikes) cudaFree(d_new_spikes);
    if (d_I_syn) cudaFree(d_I_syn);
    if (d_external) cudaFree(d_external);
    if (d_g) cudaFree(d_g);
    if (d_eligibility) cudaFree(d_eligibility);
    if (d_weight) cudaFree(d_weight);
    if (d_tau_s) cudaFree(d_tau_s);
    if (d_pre) cudaFree(d_pre);
    if (d_post) cudaFree(d_post);
}

// Explicit instantiations
template void brain_step_cuda<float>(brain<float>*, const float*, int, float, float, int, int);
template void brain_step_batch<float>(brain<float>* const*, int, const float* const*, int, float, float);

} // namespace snn
