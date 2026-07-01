#include "snake_encoder.h"
#include <cmath>
#include <cstdlib>

namespace snn {

snake_encoder* snake_encoder_create(void) {
    snake_encoder* enc = new snake_encoder;
    enc->min_rate = 0.0f;
    enc->max_rate = 100.0f;
    enc->obs_dim = 0;
    enc->sensory_neuron_indices = nullptr;
    enc->rates = nullptr;
    enc->spikes = nullptr;
    enc->use_poisson = true;
    return enc;
}

void snake_encoder_destroy(snake_encoder* enc) {
    if (!enc) return;
    delete[] enc->sensory_neuron_indices;
    delete[] enc->rates;
    delete[] enc->spikes;
    delete enc;
}

void snake_encoder_init(snake_encoder* enc,
                        brain<float>* b,
                        const brain_region_v3<float>* sensory_region,
                        int num_regions,
                        float min_rate_hz,
                        float max_rate_hz,
                        bool poisson) {
    if (!enc || !b || !sensory_region || num_regions <= 0) return;

    enc->min_rate = min_rate_hz;
    enc->max_rate = max_rate_hz;
    enc->use_poisson = poisson;
    enc->obs_dim = snake_env_get_observation_size();

    // find all neurons in the first sensory region
    int n_neurons = 0;
    brain_neuron<float>** neurons = brain_get_neurons_in_region(b, sensory_region[0], &n_neurons);
    
    enc->sensory_neuron_indices = new int[enc->obs_dim];
    enc->rates = new float[enc->obs_dim];
    enc->spikes = new float[enc->obs_dim];

    // assign first obs_dim neurons from the region as sensory inputs
    for (int i = 0; i < enc->obs_dim; ++i) {
        if (i < n_neurons) {
            enc->sensory_neuron_indices[i] = brain_get_neuron_idx(b, neurons[i]);
        } else {
            enc->sensory_neuron_indices[i] = -1;
        }
        enc->rates[i] = 0.0f;
        enc->spikes[i] = 0.0f;
    }

    delete[] neurons;
}

void snake_encoder_update(snake_encoder* enc, const float* observation) {
    if (!enc || !observation || enc->obs_dim <= 0) return;

    // map obs[i] in [0,1] to firing rate in [min_rate, max_rate]
    float dt = enc->use_poisson ? (1.0f / 1000.0f) : 1.0f; // assume 1kHz timestep

    for (int i = 0; i < enc->obs_dim; ++i) {
        float obs_val = observation[i];
        // obs values are normalized 0..1 (or -1..1 for angle)
        float normalized = obs_val * 0.5f + 0.5f; // map -1..1 to 0..1
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;
        
        enc->rates[i] = enc->min_rate + normalized * (enc->max_rate - enc->min_rate);

        if (enc->use_poisson) {
            // Poisson spike: probability = rate * dt per timestep
            float p = enc->rates[i] * dt;
            enc->spikes[i] = ((float)std::rand() / RAND_MAX < p) ? 1.0f : 0.0f;
        } else {
            // direct rate encoding: spike = rate / max_rate
            enc->spikes[i] = normalized;
        }
    }
}

void snake_encoder_apply(snake_encoder* enc, float* input_currents) {
    if (!enc || !input_currents || !enc->spikes) return;

    for (int i = 0; i < enc->obs_dim; ++i) {
        int idx = enc->sensory_neuron_indices[i];
        if (idx >= 0) {
            // spike → strong current pulse; no spike → zero
            input_currents[idx] = enc->spikes[i] * 2.0f;
        }
    }
}

} // namespace snn
