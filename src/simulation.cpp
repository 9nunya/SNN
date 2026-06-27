#include "simulation.h"
#include "collections.h"
#include "compute/cpu.h"
#include "compute/cuda.h"
#include <cmath>
#include <iostream>

namespace snn {
template<typename T>
simulation<T>* simulation_init(simulation_creation_params<T> params) {
    simulation<T>* sim = new simulation<T>;
    sim->s = new simulation_state<T>;
    sim->s->cur_time = T{};
    sim->s->inputs = nullptr;
    sim->s->outputs = nullptr;
    sim->s->v_logs = nullptr;
    sim->s->w_logs = nullptr;
    sim->s->v_th_logs = nullptr;
    sim->s->neuron_stride = 0;
    sim->s->synapse_stride = 0;
    sim->s->neurons = nullptr;
    sim->s->synapses = nullptr;
    sim->p = new simulation_params<T>;
    sim->p->backend = params.backend;
    sim->p->T_step = params.T_step;
    return sim;
}

void simulation_run_for(simulation<float>* sim, float wall_dt) {
    if (!sim || wall_dt <= 0.0f) return;

    float T_step = sim->p->T_step;
    int steps = static_cast<int>(std::ceil(wall_dt / T_step));

    int n_neurons = sim->s->neurons ? sim->s->neurons->size : 0;
    int n_synapses = sim->s->synapses ? sim->s->synapses->size : 0;
    int neuron_stride = n_neurons;
    int synapse_stride = n_synapses;
    int total_stride = neuron_stride + synapse_stride;

    if (sim->s->outputs) delete[] sim->s->outputs;
    sim->s->outputs = new float[steps * total_stride];
    sim->s->neuron_stride = neuron_stride;
    sim->s->synapse_stride = synapse_stride;

    if (sim->s->v_logs) delete[] sim->s->v_logs;
    sim->s->v_logs = n_neurons > 0 ? new float[steps * n_neurons] : nullptr;
    if (sim->s->w_logs) delete[] sim->s->w_logs;
    sim->s->w_logs = n_neurons > 0 ? new float[steps * n_neurons] : nullptr;
    if (sim->s->v_th_logs) delete[] sim->s->v_th_logs;
    sim->s->v_th_logs = n_neurons > 0 ? new float[steps * n_neurons] : nullptr;

    if (n_neurons == 0 && n_synapses == 0) {
        sim->s->cur_time += steps * T_step;
        return;
    }

    if (sim->p->backend == backend_kind::CPU) {
        for (int step = 0; step < steps; ++step) {
            float* step_out = sim->s->outputs + step * total_stride;
            const float* step_in = sim->s->inputs + step * n_neurons;
            if (n_neurons > 0) {
                int* n_idx = new int[n_neurons];
                for (int i = 0; i < n_neurons; ++i) n_idx[i] = i;
                compute_load n_load{n_idx, n_neurons};
                cpu_process_neurons(*sim->s->neurons, n_load, step_in, T_step, step_out);
                for (int i = 0; i < n_neurons; ++i) {
                    sim->s->v_logs[step * n_neurons + i] = sim->s->neurons->neurons[i]->s->v;
                    sim->s->v_th_logs[step * n_neurons + i] = sim->s->neurons->neurons[i]->s->v_th;
                    if (sim->s->neurons->neurons[i]->s->type == neuron_type::ALIF) {
                        auto *alif = static_cast<adaptive_neuron_state<float>*>(sim->s->neurons->neurons[i]->s);
                        sim->s->w_logs[step * n_neurons + i] = alif->w;
                        sim->s->v_th_logs[step * n_neurons + i] += alif->w;
                    } else {
                        sim->s->w_logs[step * n_neurons + i] = 0.0f;
                    }
                }
                delete[] n_idx;
            }
            if (n_synapses > 0) {
                int* s_idx = new int[n_synapses];
                for (int i = 0; i < n_synapses; ++i) s_idx[i] = i;
                compute_load s_load{s_idx, n_synapses};
                cpu_process_synapses(*sim->s->synapses, s_load, step_in, T_step, step_out + neuron_stride);
                delete[] s_idx;
            }
            sim->s->cur_time += T_step;
        }
    } else {
        float* d_step_out = nullptr;
        float* d_step_in = nullptr;
        int* d_n_idx = nullptr;
        int* d_s_idx = nullptr;
        if (total_stride > 0) cudaMalloc(&d_step_out, total_stride * sizeof(float));
        if (n_neurons > 0) {
            cudaMalloc(&d_step_in, n_neurons * sizeof(float));
            cudaMalloc(&d_n_idx, n_neurons * sizeof(int));
            int* h_n_idx = new int[n_neurons];
            for (int i = 0; i < n_neurons; ++i) h_n_idx[i] = i;
            cudaMemcpy(d_n_idx, h_n_idx, n_neurons * sizeof(int), cudaMemcpyHostToDevice);
            delete[] h_n_idx;
        }
        if (n_synapses > 0) {
            cudaMalloc(&d_s_idx, n_synapses * sizeof(int));
            int* h_s_idx = new int[n_synapses];
            for (int i = 0; i < n_synapses; ++i) h_s_idx[i] = i;
            cudaMemcpy(d_s_idx, h_s_idx, n_synapses * sizeof(int), cudaMemcpyHostToDevice);
            delete[] h_s_idx;
        }

        for (int step = 0; step < steps; ++step) {
            float* step_out = sim->s->outputs + step * total_stride;
            const float* step_in = sim->s->inputs + step * n_neurons;
            if (n_neurons > 0) {
                cudaMemcpy(d_step_in, step_in, n_neurons * sizeof(float), cudaMemcpyHostToDevice);
                compute_load n_load{d_n_idx, n_neurons};
                cuda_process_neurons(sim->s->neurons->device_data, n_load, d_step_in, T_step, d_step_out);
                cudaMemcpy(step_out, d_step_out, neuron_stride * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(sim->s->v_logs + step * n_neurons, sim->s->neurons->device_data.v, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(sim->s->v_th_logs + step * n_neurons, sim->s->neurons->device_data.v_th, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
                if (sim->s->neurons->device_data.w) {
                    cudaMemcpy(sim->s->w_logs + step * n_neurons, sim->s->neurons->device_data.w, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
                } else {
                    for (int i = 0; i < n_neurons; ++i) sim->s->w_logs[step * n_neurons + i] = 0.0f;
                }
                for (int i = 0; i < n_neurons; ++i) {
                    sim->s->v_th_logs[step * n_neurons + i] += sim->s->w_logs[step * n_neurons + i];
                }
            }
            if (n_synapses > 0) {
                compute_load s_load{d_s_idx, n_synapses};
                cuda_process_synapses(sim->s->synapses->device_data, s_load, d_step_in, T_step, d_step_out);
                cudaMemcpy(step_out + neuron_stride, d_step_out, synapse_stride * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        if (d_step_out) cudaFree(d_step_out);
        if (d_step_in) cudaFree(d_step_in);
        if (d_n_idx) cudaFree(d_n_idx);
        if (d_s_idx) cudaFree(d_s_idx);
        sim->s->cur_time += steps * T_step;
    }
}

template simulation<float>* simulation_init<float>(simulation_creation_params<float>);
}
