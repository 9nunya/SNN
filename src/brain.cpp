#include "brain.h"
#include "cppn.h"
#include "collections.h"
#include "compute/compute_ops.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace snn {

// ---- helpers ----

static bool point_in_region(const bv3<float>& p, const brain_region_v3<float>& r) {
    return p.x >= r.top_left.x && p.x <= r.bottom_right.x &&
           p.y >= r.top_left.y && p.y <= r.bottom_right.y &&
           p.z >= r.top_left.z && p.z <= r.bottom_right.z;
}

static float distance_sq(const bv3<float>& a, const bv3<float>& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// ---- brain_create ----

brain<float>* brain_create(brain_creation_params<float> params) {
    brain<float>* b = new brain<float>;
    b->neuron_field = new brain_neuron_field<float>;
    b->neuron_field->neurons = nullptr;
    b->neuron_field->synapses = nullptr;
    b->neuron_field->num_neurons = 0;
    b->neuron_field->num_synapses = 0;
    b->neurons = nullptr;
    b->synapses = nullptr;
    b->last_spikes = nullptr;
    b->T_step = params.T_step;
    b->T = 0.0f;
    b->backend = params.backend;
    b->recorder.recording = false;
    b->recorder.spikes = nullptr;
    b->recorder.regions = nullptr;
    b->recorder.region_offsets = nullptr;
    b->recorder.window = 0;
    b->recorder.num_regions = 0;
    b->recorder.head = 0;
    b->gpu = {};
    b->gpu.valid = false;
    return b;
}

// ---- brain_destroy ----

void brain_destroy(brain<float>* b) {
    if (!b) return;

    // stop recorder if active
    if (b->recorder.recording) {
        brain_recorder_stop(b);
    }

    // delete field entries
    if (b->neuron_field) {
        if (b->neuron_field->neurons) {
            for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
                delete b->neuron_field->neurons[i];
            }
            delete[] b->neuron_field->neurons;
        }
        if (b->neuron_field->synapses) {
            for (int i = 0; i < b->neuron_field->num_synapses; ++i) {
                delete b->neuron_field->synapses[i];
            }
            delete[] b->neuron_field->synapses;
        }
        delete b->neuron_field;
    }

    delete[] b->last_spikes;

    if (b->neurons) {
        // device arrays freed inside grow logic, but for destroy we need to free
        if (b->neurons->device_data.v) cudaFree(b->neurons->device_data.v);
        if (b->neurons->device_data.v_th) cudaFree(b->neurons->device_data.v_th);
        if (b->neurons->device_data.tau_rc) cudaFree(b->neurons->device_data.tau_rc);
        if (b->neurons->device_data.tau_ref) cudaFree(b->neurons->device_data.tau_ref);
        if (b->neurons->device_data.rest_time) cudaFree(b->neurons->device_data.rest_time);
        if (b->neurons->device_data.a) cudaFree(b->neurons->device_data.a);
        if (b->neurons->device_data.b) cudaFree(b->neurons->device_data.b);
        if (b->neurons->device_data.e) cudaFree(b->neurons->device_data.e);
        if (b->neurons->device_data.slf) cudaFree(b->neurons->device_data.slf);
        if (b->neurons->device_data.type) cudaFree(b->neurons->device_data.type);
        if (b->neurons->device_data.w) cudaFree(b->neurons->device_data.w);
        if (b->neurons->device_data.tau_w) cudaFree(b->neurons->device_data.tau_w);
        for (size_t i = 0; i < b->neurons->neurons.size(); ++i) delete b->neurons->neurons[i];
        delete b->neurons;
    }

    if (b->synapses) {
        if (b->synapses->device_data.tau_s) cudaFree(b->synapses->device_data.tau_s);
        if (b->synapses->device_data.g) cudaFree(b->synapses->device_data.g);
        if (b->synapses->device_data.weight) cudaFree(b->synapses->device_data.weight);
        if (b->synapses->device_data.E_rev) cudaFree(b->synapses->device_data.E_rev);
        if (b->synapses->device_data.eligibility) cudaFree(b->synapses->device_data.eligibility);
        if (b->synapses->device_data.type) cudaFree(b->synapses->device_data.type);
        for (size_t i = 0; i < b->synapses->synapses.size(); ++i) delete b->synapses->synapses[i];
        delete b->synapses;
    }

    if (b->gpu.d_pre) cudaFree(b->gpu.d_pre);
    if (b->gpu.d_post) cudaFree(b->gpu.d_post);
    if (b->gpu.d_old_spikes) cudaFree(b->gpu.d_old_spikes);
    if (b->gpu.d_new_spikes) cudaFree(b->gpu.d_new_spikes);
    if (b->gpu.d_I_syn) cudaFree(b->gpu.d_I_syn);

    delete b;
}

// ---- brain_populate_neurons ----

void brain_populate_neurons(brain<float>* b, brain_populate_neurons_params<float> params) {
    if (!b || params.num_neurons <= 0) return;

    int old_count = b->neuron_field ? b->neuron_field->num_neurons : 0;
    int new_count = old_count + params.num_neurons;

    // 1. Grow neuron_collection
    if (!b->neurons) {
        if (params.cppn) {
            // per-neuron CPPN hyperparameters: need adaptive neuron creation
            adaptive_neuron_creation_parameters<float> ap;
            ap.v_th = 1.0f;
            ap.tau_rc = 0.02f;
            ap.tau_ref = 0.002f;
            ap.v_init = 0.0f;
            ap.slf = false;
            ap.b = 0.0f; // default LIF-like
            ap.tau_w = 0.3f;
            ap.w_init = 0.0f;
            b->neurons = neuron_collection_init<float>(new_count, ap,
                params.max_rate_range, params.intercept_range, params.encoder_choices);
        } else {
            // build adaptive params safely from base params
            adaptive_neuron_creation_parameters<float> ap2;
            ap2.v_th = params.neuron_params.v_th;
            ap2.tau_rc = params.neuron_params.tau_rc;
            ap2.tau_ref = params.neuron_params.tau_ref;
            ap2.v_init = params.neuron_params.v_init;
            ap2.slf = params.neuron_params.slf;
            ap2.b = 0.0f;
            ap2.tau_w = 0.3f;
            ap2.w_init = 0.0f;
            b->neurons = neuron_collection_init<float>(new_count, ap2,
                params.max_rate_range, params.intercept_range, params.encoder_choices);
        }
    } else {
        // grow existing collection
        auto* old_coll = b->neurons;
        adaptive_neuron_creation_parameters<float> ap2;
        ap2.v_th = params.neuron_params.v_th;
        ap2.tau_rc = params.neuron_params.tau_rc;
        ap2.tau_ref = params.neuron_params.tau_ref;
        ap2.v_init = params.neuron_params.v_init;
        ap2.slf = params.neuron_params.slf;
        ap2.b = 0.0f;
        ap2.tau_w = 0.3f;
        ap2.w_init = 0.0f;
        auto* new_coll = neuron_collection_init<float>(new_count, ap2,
            params.max_rate_range, params.intercept_range, params.encoder_choices);

        auto& od = old_coll->device_data;
        auto& nd = new_coll->device_data;
        cudaMemcpy(nd.v, od.v, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.v_th, od.v_th, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.tau_rc, od.tau_rc, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.tau_ref, od.tau_ref, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.rest_time, od.rest_time, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.a, od.a, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.b, od.b, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.e, od.e, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.slf, od.slf, old_count * sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.type, od.type, old_count * sizeof(int), cudaMemcpyDeviceToDevice);
        if (od.w) cudaMemcpy(nd.w, od.w, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        if (od.tau_w) cudaMemcpy(nd.tau_w, od.tau_w, old_count * sizeof(float), cudaMemcpyDeviceToDevice);

        for (int i = 0; i < new_count; ++i) {
            if (i < old_count) {
                delete new_coll->neurons[i];
                new_coll->neurons[i] = old_coll->neurons[i];
            }
        }

        cudaFree(od.v); cudaFree(od.v_th); cudaFree(od.tau_rc);
        cudaFree(od.tau_ref); cudaFree(od.rest_time); cudaFree(od.a);
        cudaFree(od.b); cudaFree(od.e); cudaFree(od.slf);
        cudaFree(od.type);
        if (od.w) cudaFree(od.w);
        if (od.tau_w) cudaFree(od.tau_w);
        delete old_coll;
        b->neurons = new_coll;
    }

    // 2. Grow field->neurons array
    brain_neuron<float>** new_arr = new brain_neuron<float>*[new_count];
    if (b->neuron_field->neurons) {
        std::memcpy(new_arr, b->neuron_field->neurons, old_count * sizeof(brain_neuron<float>*));
        delete[] b->neuron_field->neurons;
    }
    b->neuron_field->neurons = new_arr;

    // 3. If CPPN provided, apply per-neuron hyperparameters to the newly created neurons
    bool use_cppn = (params.cppn != nullptr);

    for (int i = old_count; i < new_count; ++i) {
        b->neuron_field->neurons[i] = new brain_neuron<float>;
        b->neuron_field->neurons[i]->n = b->neurons->neurons[i];
        b->neuron_field->neurons[i]->last_fire_time = 0.0f;
        b->neuron_field->neurons[i]->xyz.x = util::random_uniform(params.region.top_left.x, params.region.bottom_right.x);
        b->neuron_field->neurons[i]->xyz.y = util::random_uniform(params.region.top_left.y, params.region.bottom_right.y);
        b->neuron_field->neurons[i]->xyz.z = util::random_uniform(params.region.top_left.z, params.region.bottom_right.z);

        // Override neuron params with CPPN if available
        if (use_cppn && i >= old_count) {
            float out[5];
            auto& xyz = b->neuron_field->neurons[i]->xyz;
            cppn_forward_single(*params.cppn, xyz.x, xyz.y, xyz.z, out);
            float v_th = 0.5f + out[0] * 1.0f;
            float tau_rc = 0.01f * std::exp(out[1] * 2.0f);
            float tau_ref = 0.001f * std::exp(out[2] * 2.0f);
            b->neurons->neurons[i]->s->v_th = v_th;
            b->neurons->neurons[i]->s->tau_rc = tau_rc;
            b->neurons->neurons[i]->s->tau_ref = tau_ref;
        }
    }

    // Sync CPPN-modified neuron params back to device
    if (use_cppn && params.cppn) {
        int n_new = new_count - old_count;
        if (n_new > 0) {
            std::vector<float> h_v_th(n_new), h_tau_rc(n_new), h_tau_ref(n_new);
            for (int i = old_count; i < new_count; ++i) {
                h_v_th[i - old_count] = b->neurons->neurons[i]->s->v_th;
                h_tau_rc[i - old_count] = b->neurons->neurons[i]->s->tau_rc;
                h_tau_ref[i - old_count] = b->neurons->neurons[i]->s->tau_ref;
            }
            cudaMemcpy(b->neurons->device_data.v_th + old_count, h_v_th.data(), n_new * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(b->neurons->device_data.tau_rc + old_count, h_tau_rc.data(), n_new * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(b->neurons->device_data.tau_ref + old_count, h_tau_ref.data(), n_new * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    b->neuron_field->num_neurons = new_count;

    // Grow last_spikes array
    bool* new_spikes = new bool[new_count];
    if (b->last_spikes) {
        std::memcpy(new_spikes, b->last_spikes, old_count * sizeof(bool));
        delete[] b->last_spikes;
    }
    std::fill(new_spikes + old_count, new_spikes + new_count, false);
    b->last_spikes = new_spikes;

    b->gpu.valid = false;
}

// ---- brain_create_synapse ----

void brain_create_synapse(brain<float>* b, brain_create_synapse_params<float> params) {
    if (!b) return;

    int old_count = b->neuron_field ? b->neuron_field->num_synapses : 0;
    int new_count = old_count + 1;

    // 1. Grow synapse_collection
    if (!b->synapses) {
        b->synapses = synapse_collection_init<float>(1, params.synapse_params);
    } else {
        auto* old_coll = b->synapses;
        auto* new_coll = synapse_collection_init<float>(new_count, params.synapse_params);

        auto& od = old_coll->device_data;
        auto& nd = new_coll->device_data;
        cudaMemcpy(nd.tau_s, od.tau_s, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.g, od.g, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.weight, od.weight, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.type, od.type, old_count * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.eligibility, od.eligibility, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        if (od.E_rev && nd.E_rev) {
            cudaMemcpy(nd.E_rev, od.E_rev, old_count * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        for (int i = 0; i < new_count; ++i) {
            if (i < old_count) {
                delete new_coll->synapses[i];
                new_coll->synapses[i] = old_coll->synapses[i];
            }
        }

        cudaFree(od.tau_s); cudaFree(od.g); cudaFree(od.weight);
        cudaFree(od.type);
        if (od.E_rev) cudaFree(od.E_rev);
        if (od.eligibility) cudaFree(od.eligibility);
        delete old_coll;
        b->synapses = new_coll;
    }

    // 2. Grow field->synapses array
    brain_synapse<float>** new_arr = new brain_synapse<float>*[new_count];
    if (b->neuron_field->synapses) {
        std::memcpy(new_arr, b->neuron_field->synapses, old_count * sizeof(brain_synapse<float>*));
        delete[] b->neuron_field->synapses;
    }
    b->neuron_field->synapses = new_arr;

    // 3. Create brain_synapse
    int idx = old_count;
    brain_synapse<float>* bs = new brain_synapse<float>;
    bs->pre_neuron_idx = params.pre_neuron_idx;
    bs->post_neuron_idx = params.post_neuron_idx;
    bs->weight = params.weight;
    bs->eligibility_trace = 0.0f;
    bs->s = b->synapses->synapses[idx];
    bs->s->s->weight = params.weight;

    b->neuron_field->synapses[idx] = bs;
    b->neuron_field->num_synapses = new_count;

    b->gpu.valid = false;
}

// ---- brain_wire_region ----

void brain_wire_region(brain<float>* b, brain_wire_region_params<float> params) {
    if (!b || !b->neuron_field || b->neuron_field->num_neurons == 0) return;

    int N = b->neuron_field->num_neurons;
    const float max_d_sq = params.max_distance * params.max_distance;

    // Collect pre/post candidates
    std::vector<int> pre_idx, post_idx;
    for (int i = 0; i < N; ++i) {
        if (point_in_region(b->neuron_field->neurons[i]->xyz, params.pre_region))
            pre_idx.push_back(i);
        if (point_in_region(b->neuron_field->neurons[i]->xyz, params.post_region))
            post_idx.push_back(i);
    }

    // For each pre candidate, check all post candidates within max_distance
    // For small N, brute force is fine. For large N, use spatial hash.
    std::vector<std::pair<int, int>> to_create;

    for (int pi : pre_idx) {
        auto& pa = b->neuron_field->neurons[pi]->xyz;
        for (int pj : post_idx) {
            if (pi == pj && !params.self_connections) continue;
            auto& pb = b->neuron_field->neurons[pj]->xyz;
            float d_sq = distance_sq(pa, pb);
            if (d_sq > max_d_sq) continue;
            if (params.connection_probability > 0.0f && util::random_uniform(0.0f, 1.0f) > params.connection_probability)
                continue;
            if (params.cppn) {
                float out[2];
                cppn_forward(*params.cppn, pa.x, pa.y, pa.z, pb.x, pb.y, pb.z, out);
                if (out[0] < params.cppn_link_threshold) continue;
                to_create.emplace_back(pi, pj);
            } else {
                to_create.emplace_back(pi, pj);
            }
        }
    }

    if (to_create.empty()) return;

    int old_syn_count = b->neuron_field->num_synapses;
    int final_syn_count = old_syn_count + (int)to_create.size();

    // Bulk pre-allocate synapse collection to final size
    if (!b->synapses) {
        synapse_creation_parameters<float> s_p;
        s_p.tau_s = 0.01f;
        s_p.weight = 1.0f;
        b->synapses = synapse_collection_init<float>(final_syn_count, s_p);
    } else {
        auto* old_coll = b->synapses;
        auto* new_coll = synapse_collection_init<float>(final_syn_count,
            synapse_creation_parameters<float>{0.01f, 1.0f, synapse_type::ALPHA});

        auto& od = old_coll->device_data;
        auto& nd = new_coll->device_data;
        cudaMemcpy(nd.tau_s, od.tau_s, old_syn_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.g, od.g, old_syn_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.weight, od.weight, old_syn_count * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.type, od.type, old_syn_count * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(nd.eligibility, od.eligibility, old_syn_count * sizeof(float), cudaMemcpyDeviceToDevice);
        if (od.E_rev && nd.E_rev) {
            cudaMemcpy(nd.E_rev, od.E_rev, old_syn_count * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        for (int i = 0; i < final_syn_count; ++i) {
            if (i < old_syn_count) {
                delete new_coll->synapses[i];
                new_coll->synapses[i] = old_coll->synapses[i];
            }
        }

        cudaFree(od.tau_s); cudaFree(od.g); cudaFree(od.weight);
        cudaFree(od.type);
        if (od.E_rev) cudaFree(od.E_rev);
        if (od.eligibility) cudaFree(od.eligibility);
        delete old_coll;
        b->synapses = new_coll;
    }

    // Grow field->synapses array
    brain_synapse<float>** new_arr = new brain_synapse<float>*[final_syn_count];
    if (b->neuron_field->synapses) {
        std::memcpy(new_arr, b->neuron_field->synapses, old_syn_count * sizeof(brain_synapse<float>*));
        delete[] b->neuron_field->synapses;
    }
    b->neuron_field->synapses = new_arr;

    // Create new brain_synapses
    for (int k = 0; k < (int)to_create.size(); ++k) {
        int pre = to_create[k].first;
        int post = to_create[k].second;
        float w;
        if (params.cppn) {
            auto& a = b->neuron_field->neurons[pre]->xyz;
            auto& pb = b->neuron_field->neurons[post]->xyz;
            float out[2];
            cppn_forward(*params.cppn, a.x, a.y, a.z, pb.x, pb.y, pb.z, out);
            w = std::clamp(out[1], -1.0f, 1.0f);
        } else {
            w = util::random_uniform(-1.0f, 1.0f);
        }

        brain_synapse<float>* bs = new brain_synapse<float>;
        bs->pre_neuron_idx = pre;
        bs->post_neuron_idx = post;
        bs->weight = w;
        bs->eligibility_trace = 0.0f;

        int idx = old_syn_count + k;
        bs->s = b->synapses->synapses[idx];
        bs->s->s->weight = w;

        b->neuron_field->synapses[idx] = bs;
    }

    b->neuron_field->num_synapses = final_syn_count;

    b->gpu.valid = false;
}

// ---- brain_step_cpu ----

void brain_step_cpu(brain<float>* b, const float* external_inputs, int num_inputs, float reward, float dt, int N, int S) {
    // phase 1: synapse step using previous spikes
    for (int s = 0; s < S; ++s) {
        brain_synapse<float>* bs = b->neuron_field->synapses[s];
        int pre = bs->pre_neuron_idx;
        float I = (pre < N && b->last_spikes[pre]) ? bs->weight : 0.0f;
        bs->s->s->g = compute_ops::synapse_step_impl(bs->s->s, I, dt);
    }

    // phase 2: aggregate synaptic currents to post-neurons
    float* I_syn = new float[N];
    std::fill(I_syn, I_syn + N, 0.0f);
    for (int s = 0; s < S; ++s) {
        brain_synapse<float>* bs = b->neuron_field->synapses[s];
        I_syn[bs->post_neuron_idx] += bs->s->s->g;
    }

    // phase 3: neuron step
    for (int i = 0; i < N; ++i) {
        float I_ext = (i < num_inputs) ? external_inputs[i] : 0.0f;
        float I_total = I_ext + I_syn[i];
        bool spike = compute_ops::neuron_step_impl(b->neurons->neurons[i]->s, I_total, dt);
        b->last_spikes[i] = spike;
        if (spike) {
            b->neuron_field->neurons[i]->last_fire_time = b->T;
        }
    }
    delete[] I_syn;

    // phase 4: R-STDP (always runs to decay traces)
    const float A_plus  = 0.1f;
    const float A_minus = -0.1f;
    const float decay   = 0.95f;
    const float eta     = 0.01f;

    for (int s = 0; s < S; ++s) {
        brain_synapse<float>* bs = b->neuron_field->synapses[s];
        bool pre  = b->last_spikes[bs->pre_neuron_idx];
        bool post = b->last_spikes[bs->post_neuron_idx];

        if (pre && !post)      bs->eligibility_trace += A_plus;
        else if (!pre && post) bs->eligibility_trace += A_minus;
        bs->eligibility_trace *= decay;

        if (reward != 0.0f) {
            bs->weight += reward * bs->eligibility_trace * eta;
            bs->s->s->weight = bs->weight; // sync to device
        }
    }
}

// ---- brain_step (dual backend) ----

void brain_step(brain<float>* b, const float* input_currents, int num_input_currents, float reward) {
    if (!b) return;
    int N = b->neuron_field ? b->neuron_field->num_neurons : 0;
    int S = b->neuron_field ? b->neuron_field->num_synapses : 0;
    float dt = b->T_step;

    if (N == 0 && S == 0) {
        b->T += dt;
        return;
    }

    if (b->backend == backend_kind::CPU) {
        brain_step_cpu(b, input_currents, num_input_currents, reward, dt, N, S);
    } else {
        brain_step_cuda(b, input_currents, num_input_currents, reward, dt, N, S);
    }

    // recorder phase (shared)
    if (b->recorder.recording) {
        int head = b->recorder.head % b->recorder.window;
        for (int i = 0; i < N; ++i) {
            b->recorder.spikes[head * N + i] = b->last_spikes[i];
        }
        b->recorder.head++;
    }

    b->T += dt;
}

// ---- queries ----

brain_neuron<float>* brain_get_neuron(brain<float>* b, int idx) {
    if (!b || !b->neuron_field || idx < 0 || idx >= b->neuron_field->num_neurons) return nullptr;
    return b->neuron_field->neurons[idx];
}

brain_neuron<float>* brain_get_neuron(brain<float>* b, bv3<float> pos) {
    if (!b || !b->neuron_field) return nullptr;
    for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
        auto& xyz = b->neuron_field->neurons[i]->xyz;
        if (xyz.x == pos.x && xyz.y == pos.y && xyz.z == pos.z)
            return b->neuron_field->neurons[i];
    }
    return nullptr;
}

int brain_get_neuron_idx(brain<float>* b, bv3<float> pos) {
    if (!b || !b->neuron_field) return -1;
    for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
        auto& xyz = b->neuron_field->neurons[i]->xyz;
        if (xyz.x == pos.x && xyz.y == pos.y && xyz.z == pos.z)
            return i;
    }
    return -1;
}

int brain_get_neuron_idx(brain<float>* b, brain_neuron<float>* n) {
    if (!b || !b->neuron_field || !n) return -1;
    for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
        if (b->neuron_field->neurons[i] == n) return i;
    }
    return -1;
}

brain_synapse<float>* brain_get_synapse(brain<float>* b, int idx) {
    if (!b || !b->neuron_field || idx < 0 || idx >= b->neuron_field->num_synapses) return nullptr;
    return b->neuron_field->synapses[idx];
}

brain_neuron<float>** brain_get_neurons_in_region(brain<float>* b, brain_region_v3<float> region, int* out_num_neurons) {
    if (!b || !b->neuron_field || !out_num_neurons) return nullptr;
    int count = 0;
    for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
        if (point_in_region(b->neuron_field->neurons[i]->xyz, region)) ++count;
    }
    brain_neuron<float>** result = new brain_neuron<float>*[count];
    int idx = 0;
    for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
        if (point_in_region(b->neuron_field->neurons[i]->xyz, region)) {
            result[idx++] = b->neuron_field->neurons[i];
        }
    }
    *out_num_neurons = count;
    return result;
}

brain_synapse<float>** brain_get_synapses_connected_to(brain<float>* b, int neuron_idx, int* out_num_synapses) {
    if (!b || !b->neuron_field || !out_num_synapses) return nullptr;
    int count = 0;
    for (int i = 0; i < b->neuron_field->num_synapses; ++i) {
        auto* bs = b->neuron_field->synapses[i];
        if (bs->pre_neuron_idx == neuron_idx || bs->post_neuron_idx == neuron_idx) ++count;
    }
    brain_synapse<float>** result = new brain_synapse<float>*[count];
    int idx = 0;
    for (int i = 0; i < b->neuron_field->num_synapses; ++i) {
        auto* bs = b->neuron_field->synapses[i];
        if (bs->pre_neuron_idx == neuron_idx || bs->post_neuron_idx == neuron_idx) {
            result[idx++] = bs;
        }
    }
    *out_num_synapses = count;
    return result;
}

// ---- recorder ----

void brain_recorder_start(brain<float>* b, int window, brain_region_v3<float>* regions, int num_regions) {
    if (!b || window <= 0 || num_regions <= 0) return;
    brain_recorder_stop(b); // clean up any existing

    b->recorder.recording = true;
    b->recorder.window = window;
    b->recorder.num_regions = num_regions;
    b->recorder.head = 0;
    b->recorder.regions = new brain_region_v3<float>[num_regions];
    std::memcpy(b->recorder.regions, regions, num_regions * sizeof(brain_region_v3<float>));

    // compute region_offsets: for each region, count neurons in that region
    b->recorder.region_offsets = new int[num_regions + 1];
    b->recorder.region_offsets[0] = 0;
    for (int r = 0; r < num_regions; ++r) {
        int count = 0;
        for (int i = 0; i < b->neuron_field->num_neurons; ++i) {
            if (point_in_region(b->neuron_field->neurons[i]->xyz, regions[r])) ++count;
        }
        b->recorder.region_offsets[r + 1] = b->recorder.region_offsets[r] + count;
    }
    int total = b->recorder.region_offsets[num_regions];
    if (total > 0) {
        b->recorder.spikes = new bool[total * window];
        std::memset(b->recorder.spikes, 0, total * window * sizeof(bool));
    }
}

void brain_recorder_stop(brain<float>* b) {
    if (!b) return;
    if (b->recorder.spikes) { delete[] b->recorder.spikes; b->recorder.spikes = nullptr; }
    if (b->recorder.regions) { delete[] b->recorder.regions; b->recorder.regions = nullptr; }
    if (b->recorder.region_offsets) { delete[] b->recorder.region_offsets; b->recorder.region_offsets = nullptr; }
    b->recorder.recording = false;
    b->recorder.window = 0;
    b->recorder.num_regions = 0;
    b->recorder.head = 0;
}

const bool* brain_recorder_get_spikes(brain<float>* b, int region_idx, int* out_neurons) {
    if (!b || !b->recorder.recording || region_idx < 0 || region_idx >= b->recorder.num_regions) {
        if (out_neurons) *out_neurons = 0;
        return nullptr;
    }
    int head = b->recorder.head % b->recorder.window;
    int start = b->recorder.region_offsets[region_idx];
    int width = b->recorder.region_offsets[region_idx + 1] - start;
    if (out_neurons) *out_neurons = width;
    return b->recorder.spikes + head * (b->recorder.region_offsets[b->recorder.num_regions]) + start;
}

const bool* brain_recorder_get_region_window(brain<float>* b, int region_idx, int* out_num_spikes) {
    // same as get_spikes: returns pointer at current head
    return brain_recorder_get_spikes(b, region_idx, out_num_spikes);
}

const bool* brain_recorder_get_current_tick(brain<float>* b, int region_idx, int* out_num_spikes) {
    // same as get_spikes for now
    return brain_recorder_get_spikes(b, region_idx, out_num_spikes);
}

} // namespace snn
