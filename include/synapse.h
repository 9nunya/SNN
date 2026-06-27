#ifndef SNN_SYNAPSE_H
#define SNN_SYNAPSE_H
#include <cuda_runtime.h>

/// synapse (synapse x).. (sex).. (lol!!)

namespace snn {
  enum class synapse_type {
    ALPHA,
    CONDUCTANCE,
  };

  template<typename T>
  struct synapse_state {
    T 
    tau_s, // decay rate of output voltage
    g; // output voltage
    synapse_type type = synapse_type::ALPHA;
  };

  template<typename T>
  struct conductance_synapse_state : public synapse_state<T> {
    T E_rev; // reversal potential
  };

  template<typename T>
  struct synapse_creation_parameters {
    T tau_s;
    synapse_type type = synapse_type::ALPHA;
  };

  template<typename T>
  struct conductance_synapse_creation_parameters : public synapse_creation_parameters<T> {
    T E_rev;
  };

  template<typename T>
  struct synapse {
    synapse_state<T> *s;
  };

  template<typename T>
  synapse<T>* synapse_create(synapse_creation_parameters<T> params);

  template<typename T>
  synapse<T>* synapse_create(conductance_synapse_creation_parameters<T> params);
}
#endif
