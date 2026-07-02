#ifndef SNN_COLLECTIONS_H
#define SNN_COLLECTIONS_H

#include "neuron.h"
#include "synapse.h"
#include "util.h"
#include <vector>

namespace snn {
  template<typename T>
  struct neuron_gpu_data {
    T *v, *v_th, *tau_rc, *tau_ref, *rest_time, *a, *b, *e;
    bool *slf;
    T *w, *tau_w;
    int *type;
  };

  template<typename T>
  struct synapse_gpu_data {
    T *tau_s, *g;
    T *weight;
    T *E_rev;
    T *eligibility;
    int *type;
  };

  template<typename T>
  struct neuron_collection {
    std::vector<neuron<T>*> neurons;
    int size;
    neuron_gpu_data<T> device_data;
  };

  template<typename T>
  struct synapse_collection {
    std::vector<synapse<T>*> synapses;
    int size;
    synapse_gpu_data<T> device_data;
  };

  template<typename T>
  neuron_collection<T> *neuron_collection_init(int size, neuron_creation_parameters<T> n_p, range<T> max_rate_range_, range<T> intercept_range_, std::vector<T> encoder_choices_);

  template<typename T>
  neuron_collection<T> *neuron_collection_init(int size, adaptive_neuron_creation_parameters<T> n_p, range<T> max_rate_range_, range<T> intercept_range_, std::vector<T> encoder_choices_);

  template<typename T>
  synapse_collection<T> *synapse_collection_init(int size, synapse_creation_parameters<T> s_p);

  template<typename T>
  synapse_collection<T> *synapse_collection_init(int size, conductance_synapse_creation_parameters<T> s_p);
}
#endif
