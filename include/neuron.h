#ifndef SNN_NEURON_H
#define SNN_NEURON_H

#include <vector>
#include <cuda_runtime.h>

namespace snn {
  enum class neuron_type {
    LIF, // leaky integrate and fire
    ALIF, // adaptive leaky integrate and fire
  };

  template<typename T>
  struct neuron_state {
    T
     v, // membrane potential
     v_th, // membrane potential threshold for spike
     tau_rc, // membrane potential decay constant
     tau_ref, // time to rest after fire
     rest_time; // neuron rest time until neuron can spike again
    bool slf = false; // if true, when v > v_th, v -= v_th, else, v = 0
    neuron_type type = neuron_type::LIF; // neuron type
  };

  template<typename T>
  struct adaptive_neuron_state : public neuron_state<T> {
    T
     w, // how much v_th changed
     tau_w, // how w should evolve over time
     b; // how much to increase w after neuron spike
  };

  template<typename T>
  struct neuron_parameters {
    T 
     a, // gain
     b, // bias
     e; // encoder
  };

  template<typename T>
  struct neuron {
    neuron_state<T> *s;
    neuron_parameters<T> *p;
  };

  template<typename T>
  struct neuron_creation_parameters {
    T v_th, tau_rc, tau_ref, v_init = T{};
    bool slf = false;
    neuron_type type = neuron_type::LIF;
  };

  template<typename T>
  struct adaptive_neuron_creation_parameters : public neuron_creation_parameters<T> {
    T tau_w, b, w_init = T{};
  };

  template<typename T>
  struct gain_bias {
    T gain, bias;
  };

  template<typename T>
  neuron<T>* neuron_create(neuron_creation_parameters<T> params);

  template<typename T>
  neuron<T>* neuron_create(adaptive_neuron_creation_parameters<T> params);

  template<typename T>
  __host__ __device__ T neuron_analytical_rate(neuron_state<T> *n, T I);

  template<typename T>
  gain_bias<T> neuron_get_gain_bias(neuron_state<T>* n, T max_rate, T intercept);
}

#endif
