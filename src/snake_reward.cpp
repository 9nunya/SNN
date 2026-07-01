#include "snake_reward.h"
#include <cmath>

namespace snn {

float snake_reward_standard(const snake_env* e) {
    if (!e) return 0.0f;
    if (e->dead) return -1.0f;
    if (e->score > e->prev_score) return +1.0f;
    return 0.0f;
}

float snake_reward_shaped(const snake_env* e) {
    if (!e) return 0.0f;
    if (e->dead) return -1.0f;
    if (e->score > e->prev_score) return +1.0f;
    // small shaping: +0.01 for being alive, encourages survival
    return 0.01f;
}

float snake_reward_survival(const snake_env* e) {
    if (!e) return 0.0f;
    if (e->dead) return -1.0f;
    return 0.0f; // no apple reward
}

float snake_reward_dense(const snake_env* e) {
    if (!e) return 0.0f;
    if (e->dead) return -1.0f;
    // distance-based: reward proportional to reduction in distance to apple
    float prev_dx = (float)(e->apple_x - (e->head_x - e->dir_x));
    float prev_dy = (float)(e->apple_y - (e->head_y - e->dir_y));
    float prev_dist = std::sqrt(prev_dx*prev_dx + prev_dy*prev_dy);
    
    float dx = (float)(e->apple_x - e->head_x);
    float dy = (float)(e->apple_y - e->head_y);
    float curr_dist = std::sqrt(dx*dx + dy*dy);
    
    if (prev_dist > 0.001f) {
        float improvement = (prev_dist - curr_dist) / prev_dist;
        return improvement * 0.1f; // scaled shaping
    }
    return 0.0f;
}

} // namespace snn
