#ifndef SNN_COMPUTE_COMPUTE_OPS_H
#define SNN_COMPUTE_COMPUTE_OPS_H

#include "neuron.h"
#include "synapse.h"

namespace snn::compute_ops {
template<typename T>
__host__ __device__ bool neuron_step_impl(neuron_state<T> *n, T I, T T_step);

template<typename T>
__host__ __device__ T synapse_step_impl(synapse_state<T> *s, T I, T T_step);
}

#endif
