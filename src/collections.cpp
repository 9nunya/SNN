#include "collections.h"
#include <cuda_runtime.h>

namespace snn {

void launch_full_sim(neuron_gpu_data<float> n, synapse_gpu_data<float> s, float* d_out, int size, int steps, float T_step);

template<typename T>
neuron_collection<T> *neuron_collection_init(int size, neuron_creation_parameters<T> n_p, range<T> max_rate_range_, range<T> intercept_range_, std::vector<T> encoder_choices_) {
    neuron_collection<T> *nc = new neuron_collection<T>;
    nc->size = size;
    cudaMalloc(&nc->device_data.v, size * sizeof(T));
    cudaMalloc(&nc->device_data.v_th, size * sizeof(T));
    cudaMalloc(&nc->device_data.tau_rc, size * sizeof(T));
    cudaMalloc(&nc->device_data.tau_ref, size * sizeof(T));
    cudaMalloc(&nc->device_data.rest_time, size * sizeof(T));
    cudaMalloc(&nc->device_data.a, size * sizeof(T));
    cudaMalloc(&nc->device_data.b, size * sizeof(T));
    cudaMalloc(&nc->device_data.e, size * sizeof(T));
    cudaMalloc(&nc->device_data.slf, size * sizeof(bool));

    std::vector<T> h_a(size), h_b(size), h_e(size), h_v(size, n_p.v_init), h_v_th(size, n_p.v_th), h_trc(size, n_p.tau_rc), h_trf(size, n_p.tau_ref), h_r(size, 0);
    // Use char/uint8_t for bool array to avoid std::vector<bool> pointer issues
    std::vector<uint8_t> h_slf(size, (uint8_t)n_p.slf);

    for (int i = 0; i < size; i++) {
        neuron<T> *n = new neuron<T>;
        n->s = neuron_init<T>(n_p);
        gain_bias<T> gb = neuron_get_gain_bias(n->s, util::random_uniform<T>(max_rate_range_.min, max_rate_range_.max), util::random_uniform<T>(intercept_range_.min, intercept_range_.max));
        n->p = new neuron_parameters<T>{gb.gain, gb.bias, util::random_choice<T>(encoder_choices_)};
        h_a[i] = n->p->a; h_b[i] = n->p->b; h_e[i] = n->p->e;
        nc->neurons.push_back(n);
    }

    cudaMemcpy(nc->device_data.v, h_v.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.v_th, h_v_th.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.tau_rc, h_trc.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.tau_ref, h_trf.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.rest_time, h_r.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.a, h_a.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.b, h_b.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.e, h_e.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(nc->device_data.slf, h_slf.data(), size * sizeof(bool), cudaMemcpyHostToDevice);
    return nc;
}

template<typename T>
synapse_collection<T> *synapse_collection_init(int size, T tau_s) {
    synapse_collection<T> *sc = new synapse_collection<T>;
    sc->size = size;
    cudaMalloc(&sc->device_data.g, size * sizeof(T));
    cudaMalloc(&sc->device_data.tau_s, size * sizeof(T));
    std::vector<T> h_g(size, 0), h_ts(size, tau_s);
    cudaMemcpy(sc->device_data.g, h_g.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(sc->device_data.tau_s, h_ts.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    return sc;
}

template<typename T>
void run_simulation_gpu(neuron_collection<T>* nc, synapse_collection<T>* sc, T* h_output, int steps, T T_step) {
    T* d_log;
    cudaMalloc(&d_log, nc->size * steps * sizeof(T));
    launch_full_sim(nc->device_data, sc->device_data, d_log, nc->size, steps, (float)T_step);
    cudaMemcpy(h_output, d_log, nc->size * steps * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_log);
}

template void run_simulation_gpu<float>(neuron_collection<float>*, synapse_collection<float>*, float*, int, float);
template neuron_collection<float> *neuron_collection_init<float>(int, neuron_creation_parameters<float>, range<float>, range<float>, std::vector<float>);
template synapse_collection<float> *synapse_collection_init<float>(int, float);
}