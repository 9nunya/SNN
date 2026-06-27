#include "collections.h"
#include <cuda_runtime.h>
#include <cmath>

namespace snn {

__global__ void snn_simulation_kernel(neuron_gpu_data<float> n_data, synapse_gpu_data<float> s_data, float* output_log, int num_neurons, int total_steps, float T_step) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_neurons) return;

    // Temporary objects to interface with shared neuron/synapse logic
    neuron_state<float> n;
    n.v = n_data.v[nid];
    n.v_th = n_data.v_th[nid];
    n.tau_rc = n_data.tau_rc[nid];
    n.tau_ref = n_data.tau_ref[nid];
    n.rest_time = n_data.rest_time[nid];
    n.slf = n_data.slf[nid];

    synapse_state<float> s;
    s.tau_s = s_data.tau_s[nid];
    s.g = s_data.g[nid];

    float a = n_data.a[nid];
    float b = n_data.b[nid];
    float e = n_data.e[nid];

    for (int step = 0; step < total_steps; ++step) {
        float t = step * T_step;
        float input_signal = sinf(t);
        float drive = (input_signal * a * e) + b;

        float spike = neuron_forward(&n, drive, T_step);
        float current = synapse_forward(&s, spike, T_step);

        output_log[step * num_neurons + nid] = current;
    }

    // Write final state back to global memory
    n_data.v[nid] = n.v;
    n_data.rest_time[nid] = n.rest_time;
    s_data.g[nid] = s.g;
}

void launch_full_sim(neuron_gpu_data<float> n, synapse_gpu_data<float> s, float* d_out, int size, int steps, float T_step) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    snn_simulation_kernel<<<blocks, threads>>>(n, s, d_out, size, steps, T_step);
    cudaDeviceSynchronize();
}

}