#include "neuron.h"
#include <cmath>

namespace snn {
  template<typename T>
  neuron<T>* neuron_create(neuron_creation_parameters<T> params) {
    neuron<T> *n = new neuron<T>;
    n->s = new neuron_state<T>;
    n->s->v = params.v_init;
    n->s->v_th = params.v_th;
    n->s->tau_rc = params.tau_rc;
    n->s->tau_ref = params.tau_ref;
    n->s->rest_time = 0;
    n->s->slf = params.slf;
    n->s->type = neuron_type::LIF;
    n->p = nullptr;
    return n;
  }

  template<typename T>
  neuron<T>* neuron_create(adaptive_neuron_creation_parameters<T> params) {
    neuron<T> *n = new neuron<T>;
    n->s = new adaptive_neuron_state<T>;
    n->s->v = params.v_init;
    n->s->v_th = params.v_th;
    n->s->tau_rc = params.tau_rc;
    n->s->tau_ref = params.tau_ref;
    n->s->rest_time = 0;
    n->s->slf = params.slf;
    n->s->type = neuron_type::ALIF;
    auto *alif = static_cast<adaptive_neuron_state<T>*>(n->s);
    alif->tau_w = params.tau_w;
    alif->b = params.b;
    alif->w = params.w_init;
    n->p = nullptr;
    return n;
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

  template neuron<float>* neuron_create<float>(neuron_creation_parameters<float>);
  template neuron<float>* neuron_create<float>(adaptive_neuron_creation_parameters<float>);
  template __host__ __device__ float neuron_analytical_rate<float>(neuron_state<float>*, float);
  template gain_bias<float> neuron_get_gain_bias<float>(neuron_state<float>*, float, float);
}
