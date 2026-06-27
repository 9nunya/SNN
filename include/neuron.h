%%writefile include/neuron.h
#ifndef SNN_NEURON_H
#define SNN_NEURON_H

#include <vector>
#include <cuda_runtime.h>

namespace snn {
  template<typename T>
  struct neuron_state {
    T
     v, // membrane potential
     v_th, // membrane potential threshold for spike
     tau_rc, // membrane potential decay constant
     tau_ref, // time to rest after fire
     rest_time, // neuron rest time until neuron can spike again
     w, // how much v_th changed
     tau_w, // how w should evolve over time
     b; // how much to increase w after neuron spike
    bool slf = false; // if true, when v > v_th, v -= v_th, else, v = 0
  };

  template<typename T>
  struct neuron_parameters {
    T a, b, e;
  };

  template<typename T>
  struct neuron {
    neuron_state<T> *s;
    neuron_parameters<T> *p;
  };

  template<typename T>
  struct neuron_creation_parameters {
    T v_th, tau_rc, tau_ref, v_init, tau_w, b = T{};
    bool slf = false;
  };

  template<typename T>
  struct gain_bias {
    T gain, bias;
  };

  template<typename T>
  neuron_state<T> *neuron_init(neuron_creation_parameters<T> params);

  template<typename T>
  __host__ __device__ T neuron_forward(neuron_state<T> *n, T I, T T_step);

  template<typename T>
  __host__ __device__ T neuron_analytical_rate(neuron_state<T> *n, T I);

  template<typename T>
  gain_bias<T> neuron_get_gain_bias(neuron_state<T>* n, T max_rate, T intercept);
}

#endif