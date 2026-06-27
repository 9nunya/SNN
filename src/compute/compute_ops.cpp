#include "compute/compute_ops.h"
#include <cmath>

namespace snn::compute_ops {
template<typename T>
__host__ __device__ T neuron_step_impl(neuron_state<T> *n, T I, T T_step) {
    T output = T{0};
    T dt = T_step - n->rest_time < 0 ? 0 : T_step - n->rest_time > T_step ? T_step : T_step - n->rest_time;

    if (n->rest_time > 0) {
        n->rest_time -= T_step;
    } else {
        n->v = I + (n->v - I) * std::exp(-dt / n->tau_rc);
    }

    if (n->type == neuron_type::ALIF) {
      auto *alif = static_cast<adaptive_neuron_state<T>*>(n);
      if (n->v >= n->v_th + alif->w) {
        output = T{1} / dt;
        alif->w += alif->b;
        T st = dt + n->tau_rc * std::log((n->v - I) / (n->v_th - I));
        n->rest_time = n->tau_ref + st;
        n->v = n->slf ? n->v - n->v_th : T{0};
      }
      alif->w = alif->w * std::exp(-dt / alif->tau_w) + alif->b * (output > 0);
    } else {
      if (n->v >= n->v_th) {
        output = T{1} / dt;
        T st = dt + n->tau_rc * std::log((n->v - I) / (n->v_th - I));
        n->rest_time = n->tau_ref + st;
        n->v = n->slf ? n->v - n->v_th : T{0};
      }
    }

    return output;
  }

template __host__ __device__ float neuron_step_impl<float>(neuron_state<float>*, float, float);
}

namespace snn::compute_ops {
template<typename T>
__host__ __device__ T synapse_step_impl(synapse_state<T> *s, T I, T T_step) {
    s->g = I + std::exp(-T_step / s->tau_s) * (s->g - I);
    return s->g;
}

template __host__ __device__ float synapse_step_impl<float>(synapse_state<float>*, float, float);
}
