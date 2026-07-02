#ifndef SNN_SNAKE_BRAIN_H
#define SNN_SNAKE_BRAIN_H

#include "brain.h"
#include "brain_builder.h"
#include "cppn.h"
#include "snake_env.h"
#include "snake_encoder.h"
#include "snake_reward.h"
#include "tournament_recorder.h"

namespace snn {

// The modular snake brain: self-contained unit that owns its
// environment, encoder, and SNN. Provides a clean step/action interface
// for the tournament or any other training loop.
//
// This is the standard interface for all future modular brains.
// To create a new brain type:
//   1. Define your env (like snake_env)
//   2. Define your encoder (like snake_encoder)
//   3. Define your reward functions (like snake_reward_*)
//   4. Pack them into a brain_brain struct and provide step/reset/action

struct snake_brain {
    brain<float>* brain_;         // the SNN
    snake_env    env;            // the game environment
    snake_encoder* encoder;      // observation → spike encoder
    const cppn_genome* cppn;     // the CPPN that built this brain (not owned)
    brain_builder* builder;      // the recipe used to build (not owned)
    tournament_recorder* recorder; // for emitting game frames (not owned)
    float        T_step;         // simulation timestep
    int          sim_steps;      // max steps per episode
    bool         alive;          // episode still running
    int          generation;     // which generation this brain belongs to
    int          species_id;     // which species (CPPN) this brain belongs to
    int          child_idx;      // which child within species (for recorder)
    int          step_count;     // current step within episode (for recorder throttle)
};

// Create a snake brain from a brain_builder recipe + CPPN genome.
// The builder defines cortex layout and wiring rules.
// The CPPN defines the specific topology/weights for this brain.
// If recorder is provided, emits per-frame game state for visualization.
snake_brain* snake_brain_create(brain_builder* builder,
                                const cppn_genome* cppn,
                                int sim_steps,
                                int generation,
                                int species_id,
                                tournament_recorder* recorder);

// Destroy the snake brain and all owned resources.
void snake_brain_destroy(snake_brain* sb);

// Reset the brain and environment for a new episode.
// Keeps the same brain weights/state (continues learning across episodes).
void snake_brain_reset(snake_brain* sb);

// Step the brain through one timestep of the snake game.
// Internally: get obs → encode → brain_step → decode action → env_step → reward → R-STDP
// Returns the action taken (0=up, 1=right, 2=down, 3=left) or -1 if dead.
int snake_brain_step(snake_brain* sb);

// Step the brain with an explicit external reward (for R-STDP commit).
// Used when you want to apply a reward outside the normal step loop.
void snake_brain_step_reward(snake_brain* sb, float reward);

// Get the current score from the environment.
int snake_brain_get_score(const snake_brain* sb);

// Check if the episode is done.
bool snake_brain_is_done(const snake_brain* sb);

// Get current observation from the environment (for external use).
void snake_brain_get_observation(snake_brain* sb, float* out);

// Get current reward (for external use).
float snake_brain_get_reward(const snake_brain* sb);

// Set reward function type (future: allow swapping reward functions).
// v1 uses standard reward (+1 apple, -1 death).
typedef float (*snake_reward_fn)(const snake_env* e);
void snake_brain_set_reward_fn(snake_brain* sb, snake_reward_fn fn);

}

#endif
