#ifndef SNN_SNAKE_ENV_H
#define SNN_SNAKE_ENV_H

#include <cstdint>

namespace snn {

// Simple grid snake environment. Single agent, discrete actions.
// Observation: 9 floats (direction to apple + 4 danger rays + 4 diagonal danger rays).
// Action: 0=up, 1=right, 2=down, 3=left
struct snake_env {
    int width, height;
    int head_x, head_y;
    int tail_x[256], tail_y[256];
    int tail_len;
    int dir_x, dir_y;
    int apple_x, apple_y;
    int prev_score;
    int score;
    int steps_alive;
    int max_steps;
    bool dead;
};

void snake_env_create(snake_env* e, int width, int height, int max_steps);
void snake_env_destroy(snake_env* e);
void snake_env_reset(snake_env* e);
void snake_env_step(snake_env* e, int action);
bool snake_env_is_done(const snake_env* e);
int  snake_env_get_observation_size(void);
void snake_env_get_observation(const snake_env* e, float* out);
float snake_env_get_reward(const snake_env* e);

}

#endif
