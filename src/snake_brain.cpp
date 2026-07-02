#include "snake_brain.h"
#include "tournament_recorder.h"
#include <cstdlib>
#include <cmath>

namespace snn {

snake_brain* snake_brain_create(brain_builder* builder,
                                const cppn_genome* cppn,
                                int sim_steps,
                                int generation,
                                int species_id,
                                tournament_recorder* recorder) {
    if (!builder || !cppn) return nullptr;

    snake_brain* sb = new snake_brain;
    sb->builder = builder;
    sb->cppn = cppn;
    sb->recorder = recorder;
    sb->T_step = 0.001f;
    sb->sim_steps = sim_steps;
    sb->generation = generation;
    sb->species_id = species_id;
    sb->child_idx = -1; // set by tournament
    sb->step_count = 0;
    sb->alive = true;

    // Build brain from recipe + CPPN
    sb->brain_ = brain_builder_build(builder, cppn, sb->T_step, backend_kind::CPU);
    if (!sb->brain_) {
        delete sb;
        return nullptr;
    }

    // Create encoder
    sb->encoder = snake_encoder_create();
    snake_encoder_init(sb->encoder, sb->brain_,
                       &builder->cortices[0].region, // first cortex = sensory
                       1,  // num_regions
                       10.0f,   // min_rate 10Hz
                       200.0f,  // max_rate 200Hz
                       true);   // use Poisson

    // Create env
    snake_env_create(&sb->env, 10, 10, sim_steps);

    // Start recorder on motor region for action decoding
    int motor_region_idx = builder->num_cortices - 1;
    brain_recorder_start(sb->brain_, 20,
                         &builder->cortices[motor_region_idx].region,
                         1);

    return sb;
}

void snake_brain_destroy(snake_brain* sb) {
    if (!sb) return;
    if (sb->brain_) {
        brain_recorder_stop(sb->brain_);
        brain_destroy(sb->brain_);
    }
    snake_encoder_destroy(sb->encoder);
    snake_env_destroy(&sb->env);
    delete sb;
}

void snake_brain_reset(snake_brain* sb) {
    if (!sb) return;
    snake_env_reset(&sb->env);
    sb->alive = true;
    sb->step_count = 0;

    // Reset recorder
    if (sb->brain_ && sb->brain_->recorder.recording) {
        brain_recorder_stop(sb->brain_);
        // restart with same regions
        int motor_region_idx = sb->builder->num_cortices - 1;
        brain_recorder_start(sb->brain_, 20,
                             &sb->builder->cortices[motor_region_idx].region,
                             1);
    }
}

int snake_brain_step(snake_brain* sb) {
    if (!sb || !sb->alive) return -1;

    // 1. Get observation
    float obs[9];
    snake_env_get_observation(&sb->env, obs);

    // 2. Encode observation into spike trains
    snake_encoder_update(sb->encoder, obs);

    // 3. Pack encoder spikes into input_currents
    int N = sb->brain_->neuron_field->num_neurons;
    float* inputs = new float[N];
    std::fill(inputs, inputs + N, 0.0f);
    snake_encoder_apply(sb->encoder, inputs);

    // 4. Step brain (no reward yet)
    brain_step(sb->brain_, inputs, N, 0.0f);
    delete[] inputs;

    // 5. Read motor spikes to decide action
    int motor_region_idx = sb->builder->num_cortices - 1;
    int n_motor;
    brain_neuron<float>** motor = brain_get_neurons_in_region(
        sb->brain_, sb->builder->cortices[motor_region_idx].region, &n_motor);

    int action = 0;
    int best_spikes = -1;
    for (int m = 0; m < n_motor; ++m) {
        int count;
        const bool* win = brain_recorder_get_region_window(sb->brain_, m, &count);
        int spikes = 0;
        for (int w = 0; w < std::min(20, count); ++w)
            if (win[w]) spikes++;
        if (spikes > best_spikes) { best_spikes = spikes; action = m; }
    }
    delete[] motor;

    // 6. Step environment
    snake_env_step(&sb->env, action);

    // 7. Get reward and commit R-STDP if needed
    float reward = snake_env_get_reward(&sb->env);
    if (reward != 0.0f) {
        float* zeros = new float[N];
        std::fill(zeros, zeros + N, 0.0f);
        brain_step(sb->brain_, zeros, N, reward);
        delete[] zeros;
    }

    // 8. Check if done
    if (snake_env_is_done(&sb->env)) {
        sb->alive = false;
    }

    // 9. Emit frame to recorder (throttle: every 10 steps, or on state change)
    if (sb->recorder && sb->recorder->enabled) {
        bool should_emit = (sb->step_count % 10 == 0) || !sb->alive;
        if (should_emit) {
            // collect motor spike counts for the 4 motor neurons
            bool motor_spikes[4] = {false, false, false, false};
            int motor_region_idx = sb->builder->num_cortices - 1;
            int n_motor;
            brain_neuron<float>** motor = brain_get_neurons_in_region(
                sb->brain_, sb->builder->cortices[motor_region_idx].region, &n_motor);
            for (int m = 0; m < std::min(4, n_motor); ++m) {
                int count;
                const bool* win = brain_recorder_get_region_window(sb->brain_, m, &count);
                int spikes = 0;
                for (int w = 0; w < std::min(20, count); ++w)
                    if (win[w]) spikes++;
                motor_spikes[m] = (spikes > 0);
            }
            delete[] motor;

            tournament_recorder_child_frame(sb->recorder,
                                           sb->species_id,
                                           sb->child_idx,
                                           sb->step_count,
                                           &sb->env,
                                           motor_spikes,
                                           4);
        }
    }

    sb->step_count++;

    return action;
}

void snake_brain_step_reward(snake_brain* sb, float reward) {
    if (!sb || !sb->brain_) return;
    int N = sb->brain_->neuron_field->num_neurons;
    float* zeros = new float[N];
    std::fill(zeros, zeros + N, 0.0f);
    brain_step(sb->brain_, zeros, N, reward);
    delete[] zeros;
}

int snake_brain_get_score(const snake_brain* sb) {
    return sb ? sb->env.score : 0;
}

bool snake_brain_is_done(const snake_brain* sb) {
    return sb ? !sb->alive : true;
}

void snake_brain_get_observation(snake_brain* sb, float* out) {
    if (sb && out) snake_env_get_observation(&sb->env, out);
}

float snake_brain_get_reward(const snake_brain* sb) {
    return sb ? snake_env_get_reward(&sb->env) : 0.0f;
}

void snake_brain_set_reward_fn(snake_brain* sb, snake_reward_fn fn) {
    // v1: reward is hardcoded in snake_env_get_reward.
    // Future: store fn pointer and call it instead.
    (void)sb; (void)fn;
}

} // namespace snn
