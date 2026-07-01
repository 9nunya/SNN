#ifndef SNN_SIMULATION_H
#define SNN_SIMULATION_H

#include "compute/compute.h"
#include "compute/views.h"

namespace snn {
template<typename T>
struct simulation_state {
    T cur_time;
    T* inputs;
    bool* spike_outputs;
    T* synapse_outputs;
    T* v_logs;
    T* w_logs;
    T* v_th_logs;
    int neuron_stride;
    int synapse_stride;
    neuron_collection<T>* neurons;
    synapse_collection<T>* synapses;
};

template<typename T>
struct simulation_params {
    backend_kind backend;
    T T_step;
};

template<typename T>
struct simulation {
    simulation_state<T>* s;
    simulation_params<T>* p;
};

template<typename T>
struct simulation_creation_params {
    backend_kind backend;
    T T_step;
};

template<typename T>
simulation<T>* simulation_init(simulation_creation_params<T> params);

void simulation_run_for(simulation<float>* sim, float wall_dt);
}

#endif
