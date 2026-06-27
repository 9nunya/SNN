#include "synapse.h"
#include <cmath>

namespace snn {
  template<typename T>
  synapse<T>* synapse_create(synapse_creation_parameters<T> params) {
    synapse<T> *syn = new synapse<T>;
    syn->s = new synapse_state<T>;
    syn->s->tau_s = params.tau_s;
    syn->s->g = T{};
    syn->s->type = synapse_type::ALPHA;
    return syn;
  }

  template<typename T>
  synapse<T>* synapse_create(conductance_synapse_creation_parameters<T> params) {
    synapse<T> *syn = new synapse<T>;
    syn->s = new conductance_synapse_state<T>;
    syn->s->tau_s = params.tau_s;
    syn->s->g = T{};
    syn->s->type = synapse_type::CONDUCTANCE;
    static_cast<conductance_synapse_state<T>*>(syn->s)->E_rev = params.E_rev;
    return syn;
  }

  template synapse<float>* synapse_create<float>(synapse_creation_parameters<float>);
  template synapse<float>* synapse_create<float>(conductance_synapse_creation_parameters<float>);
}
