#include "synapse.h"
#include <cmath>

namespace snn {
  template<typename T>
  synapse_state<T> *synapse_init(T tau_s_) {
    synapse_state<T> *s = new synapse_state<T>;
    s->tau_s = tau_s_;
    s->g = T{};
    return s;
  }

  template<typename T>
  __host__ __device__ T synapse_forward(synapse_state<T> *s, T I, T T_step) {
    s->g = I + std::exp(-T_step / s->tau_s) * (s->g - I);
    return s->g;
  }

  // Explicit template instantiations for the linker
  template synapse_state<float>* synapse_init<float>(float);
  template __host__ __device__ float synapse_forward<float>(synapse_state<float>*, float, float);
}