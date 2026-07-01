#ifndef SNN_SNAKE_ENCODER_H
#define SNN_SNAKE_ENCODER_H

#include "brain.h"
#include "snake_env.h"

namespace snn {

// Poisson spike encoder: converts continuous observations into
// spike trains that drive the sensory cortex.
//
// v1: rate encoding. Each obs dim maps to one sensory neuron.
//      obs[i] in [0,1] → firing rate in [min_rate, max_rate] Hz
//      spike probability per timestep = rate * dt
//
// Future: full grid encoding where each cell in the snake grid
// has a corresponding neuron population, creating a viewable
// "scene" of spikes.

struct snake_encoder {
    float min_rate;   // Hz, minimum firing rate
    float max_rate;   // Hz, maximum firing rate
    int   obs_dim;    // observation dimension (should match snake_env_get_observation_size)
    int*  sensory_neuron_indices; // [obs_dim] global brain neuron indices for sensory cortex
    float* rates;     // [obs_dim] current firing rates (Hz)
    float* spikes;    // [obs_dim] current step spikes (0 or 1)
    bool  use_poisson; // if true, stochastic spikes; if false, direct rate-to-current
};

snake_encoder* snake_encoder_create(void);
void snake_encoder_destroy(snake_encoder* enc);

// Call once after brain is built: finds neurons in sensory_region
// and assigns them as the sensory neuron indices.
void snake_encoder_init(snake_encoder* enc,
                        brain<float>* b,
                        const brain_region_v3<float>* sensory_region,
                        int num_regions,
                        float min_rate_hz,
                        float max_rate_hz,
                        bool poisson);

// Convert observation vector into spike trains.
// Writes into `spikes` array (size obs_dim). Caller reads spikes
// and injects them as input_currents into the brain.
void snake_encoder_update(snake_encoder* enc, const float* observation);

// Convenience: pack encoder spikes directly into an input_currents array
// for brain_step. Assumes input_currents is zeroed.
void snake_encoder_apply(snake_encoder* enc, float* input_currents);

}

#endif
