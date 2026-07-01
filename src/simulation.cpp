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
    sim->s->spike_outputs = nullptr;
    sim->s->synapse_outputs = nullptr;
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
    sim->s->neuron_stride = n_neurons;
    sim->s->synapse_stride = n_synapses;

    if (sim->s->spike_outputs) delete[] sim->s->spike_outputs;
    sim->s->spike_outputs = n_neurons > 0 ? new bool[steps * n_neurons] : nullptr;
    if (sim->s->synapse_outputs) delete[] sim->s->synapse_outputs;
    sim->s->synapse_outputs = n_synapses > 0 ? new float[steps * n_synapses] : nullptr;

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
            const float* step_in = sim->s->inputs + step * n_neurons;
            if (n_neurons > 0) {
                int* n_idx = new int[n_neurons];
                for (int i = 0; i < n_neurons; ++i) n_idx[i] = i;
                compute_load n_load{n_idx, n_neurons};
                cpu_process_neurons(*sim->s->neurons, n_load, step_in, T_step, sim->s->spike_outputs + step * n_neurons);
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
                cpu_process_synapses(*sim->s->synapses, s_load, step_in, T_step, sim->s->synapse_outputs + step * n_synapses);
                delete[] s_idx;
            }
            sim->s->cur_time += T_step;
        }
    } else {
        float* d_step_in = nullptr;
        int* d_n_idx = nullptr;
        int* d_s_idx = nullptr;
        bool* d_spike_out = nullptr;
        float* d_syn_out = nullptr;
        if (n_neurons > 0) {
            cudaMalloc(&d_step_in, n_neurons * sizeof(float));
            cudaMalloc(&d_n_idx, n_neurons * sizeof(int));
            cudaMalloc(&d_spike_out, n_neurons * sizeof(bool));
            int* h_n_idx = new int[n_neurons];
            for (int i = 0; i < n_neurons; ++i) h_n_idx[i] = i;
            cudaMemcpy(d_n_idx, h_n_idx, n_neurons * sizeof(int), cudaMemcpyHostToDevice);
            delete[] h_n_idx;
        }
        if (n_synapses > 0) {
            cudaMalloc(&d_s_idx, n_synapses * sizeof(int));
            cudaMalloc(&d_syn_out, n_synapses * sizeof(float));
            int* h_s_idx = new int[n_synapses];
            for (int i = 0; i < n_synapses; ++i) h_s_idx[i] = i;
            cudaMemcpy(d_s_idx, h_s_idx, n_synapses * sizeof(int), cudaMemcpyHostToDevice);
            delete[] h_s_idx;
        }

        for (int step = 0; step < steps; ++step) {
            const float* step_in = sim->s->inputs + step * n_neurons;
            if (n_neurons > 0) {
                cudaMemcpy(d_step_in, step_in, n_neurons * sizeof(float), cudaMemcpyHostToDevice);
                compute_load n_load{d_n_idx, n_neurons};
                cuda_process_neurons(sim->s->neurons->device_data, n_load, d_step_in, T_step, d_spike_out);
                cudaMemcpy(sim->s->spike_outputs + step * n_neurons, d_spike_out, n_neurons * sizeof(bool), cudaMemcpyDeviceToHost);
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
                cuda_process_synapses(sim->s->synapses->device_data, s_load, d_step_in, T_step, d_syn_out);
                cudaMemcpy(sim->s->synapse_outputs + step * n_synapses, d_syn_out, n_synapses * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
        if (d_spike_out) cudaFree(d_spike_out);
        if (d_syn_out) cudaFree(d_syn_out);
        if (d_step_in) cudaFree(d_step_in);
        if (d_n_idx) cudaFree(d_n_idx);
        if (d_s_idx) cudaFree(d_s_idx);
        sim->s->cur_time += steps * T_step;
    }
}

template simulation<float>* simulation_init<float>(simulation_creation_params<float>);
}
