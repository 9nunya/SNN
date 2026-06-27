#include "cpu.h"
#include <cmath>

namespace snn {
template<typename T>
void cpu_process_neurons(neuron_collection<T>& coll, compute_load load, const T* inputs, T T_step, T* outputs) {
    for (int i = 0; i < load.count; ++i) {
        int idx = load.idx[i];
        neuron_state<T>* state = coll.neurons[idx]->s;
        outputs[idx] = compute_ops::neuron_step_impl(state, inputs[idx], T_step);
    }
}

template<typename T>
void cpu_process_synapses(synapse_collection<T>& coll, compute_load load, const T* inputs, T T_step, T* outputs) {
    for (int i = 0; i < load.count; ++i) {
        int idx = load.idx[i];
        synapse_state<T>* state = coll.synapses[idx]->s;
        outputs[idx] = compute_ops::synapse_step_impl(state, inputs[idx], T_step);
    }
}

template void cpu_process_neurons<float>(neuron_collection<float>&, compute_load, const float*, float, float*);
template void cpu_process_synapses<float>(synapse_collection<float>&, compute_load, const float*, float, float*);
}
