%%writefile include/collections.h
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
  };

  template<typename T>
  struct synapse_gpu_data {
    T *tau_s, *g;
  };

  template<typename T>
  struct neuron_collection {
    std::vector<neuron<T>*> neurons;
    int size;
    neuron_gpu_data<T> device_data;
  };

  template<typename T>
  struct synapse_collection {
    std::vector<synapse_state<T>*> synapses;
    int size;
    synapse_gpu_data<T> device_data;
  };

  template<typename T>
  neuron_collection<T> *neuron_collection_init(int size, neuron_creation_parameters<T> n_p, range<T> max_rate_range_, range<T> intercept_range_, std::vector<T> encoder_choices_);

  template<typename T>
  synapse_collection<T> *synapse_collection_init(int size, T tau_s);

  template<typename T>
  void run_simulation_gpu(neuron_collection<T>* nc, synapse_collection<T>* sc, T* h_output_buffer, int steps, T T_step);
}
#endif