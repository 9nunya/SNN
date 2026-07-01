#ifndef SNN_COMPUTE_CUDA_H
#define SNN_COMPUTE_CUDA_H

#include "collections.h"
#include "views.h"

namespace snn {
template<typename T>
void cuda_process_neurons(neuron_gpu_data<T>& data, compute_load load, const T* d_inputs, T T_step, bool* d_outputs);

template<typename T>
void cuda_process_synapses(synapse_gpu_data<T>& data, compute_load load, const T* d_inputs, T T_step, T* d_outputs);
}

#endif
