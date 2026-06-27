#include "cuda.h"
#include "compute_ops.h"
#include <cuda_runtime.h>
#include <cmath>

namespace snn {

template<typename T>
__global__ void cuda_neuron_kernel(neuron_gpu_data<T> n_data, compute_load load, const T* d_inputs, T T_step, T* d_outputs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= load.count) return;

    int nid = load.idx[i];
    int ntype = n_data.type ? n_data.type[nid] : 0;

    alignas(adaptive_neuron_state<T>) char nbuf[sizeof(adaptive_neuron_state<T>)];
    neuron_state<T> *pn = reinterpret_cast<neuron_state<T>*>(nbuf);

    pn->v = n_data.v[nid];
    pn->v_th = n_data.v_th[nid];
    pn->tau_rc = n_data.tau_rc[nid];
    pn->tau_ref = n_data.tau_ref[nid];
    pn->rest_time = n_data.rest_time[nid];
    pn->slf = n_data.slf[nid];
    pn->type = static_cast<neuron_type>(ntype);

    if (pn->type == neuron_type::ALIF) {
        auto *alif = reinterpret_cast<adaptive_neuron_state<T>*>(nbuf);
        alif->w = n_data.w ? n_data.w[nid] : T{};
        alif->tau_w = n_data.tau_w ? n_data.tau_w[nid] : T{};
        alif->b = n_data.b ? n_data.b[nid] : T{};
    }

    T input = d_inputs ? d_inputs[nid] : T{};
    T spike = compute_ops::neuron_step_impl(pn, input, T_step);

    n_data.v[nid] = pn->v;
    n_data.rest_time[nid] = pn->rest_time;
    if (d_outputs) d_outputs[nid] = spike;

    if (pn->type == neuron_type::ALIF && n_data.w) {
        auto *alif = reinterpret_cast<adaptive_neuron_state<T>*>(nbuf);
        n_data.w[nid] = alif->w;
        n_data.tau_w[nid] = alif->tau_w;
    }
}

template<typename T>
__global__ void cuda_synapse_kernel(synapse_gpu_data<T> s_data, compute_load load, const T* d_inputs, T T_step, T* d_outputs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= load.count) return;

    int sid = load.idx[i];
    int stype = s_data.type ? s_data.type[sid] : 0;

    alignas(conductance_synapse_state<T>) char sbuf[sizeof(conductance_synapse_state<T>)];
    synapse_state<T> *ps = reinterpret_cast<synapse_state<T>*>(sbuf);

    ps->tau_s = s_data.tau_s[sid];
    ps->g = s_data.g[sid];
    ps->type = static_cast<synapse_type>(stype);

    if (ps->type == synapse_type::CONDUCTANCE) {
        auto *cond = reinterpret_cast<conductance_synapse_state<T>*>(sbuf);
        cond->E_rev = s_data.E_rev ? s_data.E_rev[sid] : T{};
    }

    T input = d_inputs ? d_inputs[sid] : T{};
    T current = compute_ops::synapse_step_impl(ps, input, T_step);

    s_data.g[sid] = ps->g;
    if (d_outputs) d_outputs[sid] = current;
}

template<typename T>
void cuda_process_neurons(neuron_gpu_data<T>& data, compute_load load, const T* d_inputs, T T_step, T* d_outputs) {
    int threads = 256;
    int blocks = (load.count + threads - 1) / threads;
    cuda_neuron_kernel<<<blocks, threads>>>(data, load, d_inputs, T_step, d_outputs);
    cudaDeviceSynchronize();
}

template<typename T>
void cuda_process_synapses(synapse_gpu_data<T>& data, compute_load load, const T* d_inputs, T T_step, T* d_outputs) {
    int threads = 256;
    int blocks = (load.count + threads - 1) / threads;
    cuda_synapse_kernel<<<blocks, threads>>>(data, load, d_inputs, T_step, d_outputs);
    cudaDeviceSynchronize();
}

template void cuda_process_neurons<float>(neuron_gpu_data<float>&, compute_load, const float*, float, float*);
template void cuda_process_synapses<float>(synapse_gpu_data<float>&, compute_load, const float*, float, float*);
}
