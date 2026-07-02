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
    void brain_gpu_cache_build(brain<T>* b) {
        if (!b || !b->neuron_field) return;
        brain_gpu_cache_destroy(b->gpu);
        brain_gpu_cache_build(b->gpu, b);
    }

    template<typename T>
    void brain_gpu_cache_destroy(brain<T>* b) {
        if (!b) return;
        brain_gpu_cache_destroy(b->gpu);
    }

    template<typename T>
    void brain_gpu_cache_build(brain_gpu_cache& g, brain<T>* b) {
        g.N = b->neuron_field->num_neurons;
        g.S = b->neuron_field->num_synapses;
        if (g.S > 0) {
            cudaMalloc(&g.d_pre, g.S * sizeof(int));
            cudaMalloc(&g.d_post, g.S * sizeof(int));
            int* h_pre = new int[g.S];
            int* h_post = new int[g.S];
            for (int i = 0; i < g.S; ++i) {
                auto* bs = b->neuron_field->synapses[i];
                h_pre[i] = bs->pre_neuron_idx;
                h_post[i] = bs->post_neuron_idx;
            }
            cudaMemcpy(g.d_pre, h_pre, g.S * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(g.d_post, h_post, g.S * sizeof(int), cudaMemcpyHostToDevice);
            delete[] h_pre;
            delete[] h_post;
        }
        if (g.N > 0) {
            cudaMalloc(&g.d_old_spikes, g.N * sizeof(bool));
            cudaMalloc(&g.d_new_spikes, g.N * sizeof(bool));
            cudaMalloc(&g.d_I_syn, g.N * sizeof(float));
        }
        g.valid = true;
    }

    template<typename T>
    void brain_gpu_cache_destroy(brain_gpu_cache& g) {
        if (g.d_pre) cudaFree(g.d_pre);
        if (g.d_post) cudaFree(g.d_post);
        if (g.d_old_spikes) cudaFree(g.d_old_spikes);
        if (g.d_new_spikes) cudaFree(g.d_new_spikes);
        if (g.d_I_syn) cudaFree(g.d_I_syn);
        g = brain_gpu_cache{};
    }

}

#endif
