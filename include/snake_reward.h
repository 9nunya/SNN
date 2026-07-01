#ifndef SNN_SNAKE_REWARD_H
#define SNN_SNAKE_REWARD_H

#include "snake_env.h"

namespace snn {

// Modular reward functions for snake brain training.
// Each reward function takes the env state and returns a float reward.

// Standard reward: +1 for apple, -1 for death, 0 otherwise
float snake_reward_standard(const snake_env* e);

// Shaped reward: small positive reward for moving toward apple,
// +1 for apple, -1 for death. Helps early training.
float snake_reward_shaped(const snake_env* e);

// Survival-only: only penalizes death, no apple reward.
// Forces brain to learn avoidance first.
float snake_reward_survival(const snake_env* e);

// Dense shaping: continuous reward based on distance to apple.
float snake_reward_dense(const snake_env* e);

} // namespace snn

#endif
