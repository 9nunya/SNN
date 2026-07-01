#ifndef SNN_BRAIN_BUILDER_H
#define SNN_BRAIN_BUILDER_H

#include "brain.h"
#include "cppn.h"

namespace snn {

// One cortex definition: zone + neuron params + population count
struct builder_cortex {
    brain_region_v3<float> region;
    int num_neurons;
    neuron_creation_parameters<float> neuron_params;
    range<float> max_rate_range;
    range<float> intercept_range;
    std::vector<float> encoder_choices;
};

// One wiring rule: pre cortex -> post cortex with spatial + CPPN constraints
struct builder_wire_rule {
    int pre_cortex_idx;   // index into builder::cortices
    int post_cortex_idx;
    float max_distance;
    float cppn_link_threshold;
};

// Builds a brain from a CPPN genome according to a user-defined recipe.
// Consumer defines cortex layout + wiring rules once, then calls build
// for each species with its own CPPN.
struct brain_builder {
    builder_cortex* cortices;
    int num_cortices;
    builder_wire_rule* wire_rules;
    int num_wire_rules;
};

brain_builder* brain_builder_create(void);
void brain_builder_destroy(brain_builder* bb);

void brain_builder_add_cortex(brain_builder* bb, const builder_cortex* cortex);
void brain_builder_add_wire_rule(brain_builder* bb, const builder_wire_rule* rule);

// Builds and returns a fully populated, wired brain from the CPPN.
// Caller must brain_destroy() the returned brain when done.
// dont let it sit there.
brain<float>* brain_builder_build(brain_builder* bb,
                                  const cppn_genome* cppn,
                                  float T_step,
                                  backend_kind backend);

}

#endif
