#ifndef SNN_COMPUTE_CPU_H
#define SNN_COMPUTE_CPU_H

#include "collections.h"
#include "views.h"
#include "compute_ops.h"

namespace snn {
template<typename T>
void cpu_process_neurons(neuron_collection<T>& coll, compute_load load, const T* inputs, T T_step, T* outputs);

template<typename T>
void cpu_process_synapses(synapse_collection<T>& coll, compute_load load, const T* inputs, T T_step, T* outputs);
}

#endif
