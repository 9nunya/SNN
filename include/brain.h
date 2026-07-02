#ifndef SNN_BRAIN_H
#define SNN_BRAIN_H

#include "collections.h"
#include "compute/compute.h"
#include "cppn.h"
#include "neuron.h"
#include "synapse.h"

namespace snn {
    template<typename T>
    struct bv3 { // REVOLUTIONARY new technology.. INTRODUCING.. THE BRAIN-VEC-3!!! 
                 // wait till I make.. brainz.. in.. v4.. OMGZ..                                                       (or in v5)
        T x, y, z;
    };

    template<typename T>
    struct brain_region_v3 {
        bv3<T> top_left, bottom_right;
    };

    template<typename T>
    struct brain_neuron { // A single neuron in the brain. Stores 3d pos, neuron, and last fire time (for R-STDP)
        neuron<T> *n;
        T last_fire_time;
        bv3 xyz;
    };

    template<typename T>
    struct brain_synapse { // A single synapse in the brain. Stores pre/post neuron idx, synapse, and weight
        synapse<T> *s;
        int pre_neuron_idx;
        int post_neuron_idx;
        T weight;
        T eligibility_trace;
    };

    template<typename T>
    struct brain_neuron_field { // Represents a field of neurons that are interconnected by synapses.
        brain_neuron<T>** neurons;
        brain_synapse<T>** synapses;
        int num_neurons;
        int num_synapses;
    };

    template<typename T>
    struct brain_gpu_cache {
        bool valid = false;
        int N = 0;
        int S = 0;
        int *d_pre = nullptr;
        int *d_post = nullptr;
        bool *d_old_spikes = nullptr;
        bool *d_new_spikes = nullptr;
        float *d_I_syn = nullptr;
    };

    template<typename T>
    struct brain { // A LITERAL brain lol.
        brain_neuron_field<T> *neuron_field; // the entire consciousness of this brain
        neuron_collection<T> *neurons; // .. and it's raw neurons pointers..
        synapse_collection<T> *synapses; // .. and it's raw synapse pointers..

        T T_step;
        T T; // T_T.. you didnt know this meant the time? BRUH.

        bool *last_spikes; // [N] spikes from previous step

        // the brain recorder so we can get important stuffs!
        struct {
            bool recording;
            int window; // number of steps to record
            brain_region_v3<T> *regions; // regions to record spikes from
            int num_regions;
            bool *spikes;         // spikes is [total_region_neurons * window], flattened.
            int *region_offsets;  // region_offsets[r] gives the starting neuron index in the flat array
                                  // for region r; region_offsets[num_regions] is the total width.
            int head;             // current write head in ring buffer
        } recorder;

        backend_kind backend; // backend that the brains simmulation is running on

        brain_gpu_cache<T> gpu;
    };

    template<typename T>
    struct brain_creation_params {
        T T_step;
        backend_kind backend;
    };

    template<typename T>
    struct brain_populate_neurons_params {
        int num_neurons;
        neuron_creation_parameters<T> neuron_params;
        brain_region_v3<T> region;
        range<T> max_rate_range, intercept_range;
        std::vector<T> encoder_choices;
        const cppn_genome *cppn; // optional CPPN for per-neuron hyperparameters (nullptr = uniform random)
    };

    template<typename T>
    struct brain_create_synapse_params {
        int pre_neuron_idx, post_neuron_idx;
        synapse_creation_parameters<T> synapse_params;
        T weight;
    };

    template<typename T>
    struct brain_wire_region_params {
        brain_region_v3<T> pre_region;
        brain_region_v3<T> post_region;
        T max_distance;
        T connection_probability;
        T cppn_link_threshold;
        const cppn_genome *cppn;
    };

    template<typename T>
    brain<T>* brain_create(brain_creation_params<T> params);

    template<typename T>
    void brain_destroy(brain<T>* b);

    template<typename T>
    void brain_populate_neurons(brain<T>* b, brain_populate_neurons_params<T> params);

    template<typename T>
    void brain_create_synapse(brain<T>* b, brain_create_synapse_params<T> params);

    template<typename T>
    brain_neuron<T>* brain_get_neuron(brain<T>* b, int idx);

    template<typename T>
    brain_neuron<T>* brain_get_neuron(brain<T>* b, bv3<T> pos);

    template<typename T>
    int brain_get_neuron_idx(brain<T>* b, bv3<T> pos);

    template<typename T>
    int brain_get_neuron_idx(brain<T>* b, brain_neuron<T>* n);

    template<typename T>
    brain_synapse<T>* brain_get_synapse(brain<T>* b, int idx);

    template<typename T>
    brain_neuron<T>** brain_get_neurons_in_region(brain<T>* b, brain_region_v3<T> region, int* out_num_neurons);

    template<typename T>
    brain_synapse<T>** brain_get_synapses_connected_to(brain<T>* b, int neuron_idx, int* out_num_synapses);

    template<typename T>
    void brain_recorder_start(brain<T>* b, int window, brain_region_v3<T>* regions, int num_regions);

    template<typename T>
    void brain_recorder_stop(brain<T>* b);

    // returns flat pointer into ring buffer for region r at current head
    // spike layout: [region_offsets[r] + head * region_width + local_idx]
    const bool* brain_recorder_get_spikes(brain<T>* b, int region_idx, int* out_neurons);

    template<typename T>
    const bool* brain_recorder_get_region_window(brain<T>* b, int region_idx, int* out_num_spikes);

    template<typename T>
    const bool* brain_recorder_get_current_tick(brain<T>* b, int region_idx, int* out_num_spikes);

    template<typename T>
    void brain_step(brain<T>* b, const T* input_currents, int num_input_currents, T reward);

    template<typename T>
    void brain_wire_region(brain<T>* b, brain_wire_region_params<T> params);

    template<typename T>
    void brain_step_cpu(brain<T>* b, const T* external_inputs, int num_inputs, T reward, T dt, int N, int S);

    template<typename T>
    void brain_step_cuda(brain<T>* b, const T* external_inputs, int num_inputs, T reward, T dt, int N, int S);

    template<typename T>
    void brain_step_batch(brain<T>* const* brains, int num_brains,
                          const T* const* external_inputs, int num_inputs,
                          T reward, T dt);

    template<typename T>
    void brain_gpu_cache_build(brain_gpu_cache<T>& g, brain<T>* b);

    template<typename T>
    void brain_gpu_cache_destroy(brain_gpu_cache<T>& g);

}

#endif
