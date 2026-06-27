%%writefile src/neuron.cpp
#include "neuron.h"
#include <cmath>

namespace snn {
  template<typename T>
  neuron_state<T> *neuron_init(neuron_creation_parameters<T> params) {
    neuron_state<T> *n = new neuron_state<T>;
    n->v = params.v_init;
    n->v_th = params.v_th;
    n->tau_rc = params.tau_rc;
    n->tau_ref = params.tau_ref;
    n->rest_time = 0;
    n->slf = params.slf;
    n->tau_w = params.tau_w;
    n->b = params.b;
    n->w = 0;
    return n;
  }

  template<typename T>
  __host__ __device__ T neuron_forward(neuron_state<T> *n, T I, T T_step) {
    T output = T{0};
    T dt = T_step - n->rest_time < 0 ? 0 : T_step - n->rest_time > T_step ? T_step : T_step - n->rest_time;

    if (n->rest_time > 0) {
        n->rest_time -= T_step;
    } else {
        n->v = I + (n->v - I) * std::exp(-dt / n->tau_rc);
    }

    if (n->v >= n->v_th + n->w) {
      output = T{1} / dt;
      n->w += n->b;
      T st = dt + n->tau_rc * std::log((n->v - I) / (n->v_th - I));
      n->rest_time = n->tau_ref + st;
      n->v = n->slf ? n->v - n->v_th : T{0};
    }

    n->w = n->w * std::exp(-dt / n->tau_w) + n->b * (output > 0);
    return output;
  }

  template<typename T>
  __host__ __device__ T neuron_analytical_rate(neuron_state<T> *n, T I) {
    if (I <= n->v_th) return T{};
    else return T{1} / (n->tau_ref - n->tau_rc * log(1 - n->v_th/I));
  }

  template<typename T>
  gain_bias<T> neuron_get_gain_bias(neuron_state<T>* n, T max_rate, T intercept) {
    T term = (n->tau_ref - T{1}/max_rate) / n->tau_rc;
    T denom = T{1} - (T{1} / (T{1} - exp(term)));
    gain_bias<T> gb;
    gb.gain = (n->v_th * denom) / (intercept - T{1});
    gb.bias = n->v_th - gb.gain * intercept;
    return gb;
  }

  // Explicit instantiations ensure the compiler generates code for these types
  // so that the GPU linker can find them across translation units.
  template neuron_state<float>* neuron_init<float>(neuron_creation_parameters<float>);
  template __host__ __device__ float neuron_forward<float>(neuron_state<float>*, float, float);
  template __host__ __device__ float neuron_analytical_rate<float>(neuron_state<float>*, float);
  template gain_bias<float> neuron_get_gain_bias<float>(neuron_state<float>*, float, float);
}