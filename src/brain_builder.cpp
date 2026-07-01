#include "brain_builder.h"
#include <cstring>
#include <cmath>

namespace snn {

brain_builder* brain_builder_create(void) {
    brain_builder* bb = new brain_builder;
    bb->cortices = nullptr;
    bb->num_cortices = 0;
    bb->wire_rules = nullptr;
    bb->num_wire_rules = 0;
    return bb;
}

void brain_builder_destroy(brain_builder* bb) {
    if (!bb) return;
    delete[] bb->cortices;
    delete[] bb->wire_rules;
    delete bb;
}

void brain_builder_add_cortex(brain_builder* bb, const builder_cortex* cortex) {
    if (!bb || !cortex) return;
    int old = bb->num_cortices;
    bb->num_cortices++;
    builder_cortex* arr = new builder_cortex[bb->num_cortices];
    if (old > 0) {
        std::memcpy(arr, bb->cortices, old * sizeof(builder_cortex));
        delete[] bb->cortices;
    }
    arr[old] = *cortex;
    bb->cortices = arr;
}

void brain_builder_add_wire_rule(brain_builder* bb, const builder_wire_rule* rule) {
    if (!bb || !rule) return;
    int old = bb->num_wire_rules;
    bb->num_wire_rules++;
    builder_wire_rule* arr = new builder_wire_rule[bb->num_wire_rules];
    if (old > 0) {
        std::memcpy(arr, bb->wire_rules, old * sizeof(builder_wire_rule));
        delete[] bb->wire_rules;
    }
    arr[old] = *rule;
    bb->wire_rules = arr;
}

brain<float>* brain_builder_build(brain_builder* bb,
                                  const cppn_genome* cppn,
                                  float T_step,
                                  backend_kind backend) {
    if (!bb) return nullptr;

    brain_creation_params<float> bp;
    bp.T_step = T_step;
    bp.backend = backend;
    brain<float>* b = brain_create(bp);

    // Phase 1: populate all cortices
    for (int c = 0; c < bb->num_cortices; ++c) {
        brain_populate_neurons_params<float> pp;
        pp.num_neurons = bb->cortices[c].num_neurons;
        pp.neuron_params = bb->cortices[c].neuron_params;
        pp.region = bb->cortices[c].region;
        pp.max_rate_range = bb->cortices[c].max_rate_range;
        pp.intercept_range = bb->cortices[c].intercept_range;
        pp.encoder_choices = bb->cortices[c].encoder_choices;
        pp.cppn = cppn; // optional per-neuron hyperparams from CPPN
        brain_populate_neurons(b, pp);
    }

    // Phase 2: wire according to rules
    for (int r = 0; r < bb->num_wire_rules; ++r) {
        brain_wire_region_params<float> wp;
        wp.pre_region = bb->cortices[bb->wire_rules[r].pre_cortex_idx].region;
        wp.post_region = bb->cortices[bb->wire_rules[r].post_cortex_idx].region;
        wp.max_distance = bb->wire_rules[r].max_distance;
        wp.connection_probability = 0.0f;
        wp.cppn_link_threshold = bb->wire_rules[r].cppn_link_threshold;
        wp.cppn = cppn;
        wp.synapse_params = { 0.01f, 1.0f, synapse_type::ALPHA };
        brain_wire_region(b, wp);
    }

    return b;
}

} // namespace snn
