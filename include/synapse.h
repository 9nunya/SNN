#ifndef SNN_SYNAPSE_H
#define SNN_SYNAPSE_H
#include <cuda_runtime.h>

/// synapse (synapse x).. (sex).. (lol!!)

namespace snn {
  template<typename T>
  struct synapse_state {
    T tau_s, g;
  };

  template<typename T>
  synapse_state<T> *synapse_init(T tau_s_);

  template<typename T>
  __host__ __device__ T synapse_forward(synapse_state<T> *s, T I, T T_step);
}
#endif