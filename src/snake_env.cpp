#include "snake_env.h"
#include <cstdlib>
#include <cmath>

namespace snn {

static int rand_int(int lo, int hi) {
    return lo + std::rand() % (hi - lo);
}

static bool cell_occupied(const snake_env* e, int x, int y, bool include_head) {
    if (!include_head) {
        if (x == e->head_x && y == e->head_y) return true;
    }
    for (int i = 0; i < e->tail_len; ++i) {
        if (e->tail_x[i] == x && e->tail_y[i] == y) return true;
    }
    return false;
}

void snake_env_create(snake_env* e, int width, int height, int max_steps) {
    if (!e) return;
    e->width = width;
    e->height = height;
    e->max_steps = max_steps;
    snake_env_reset(e);
}

void snake_env_destroy(snake_env* e) {
    (void)e;
}

void snake_env_reset(snake_env* e) {
    if (!e) return;
    e->head_x = e->width / 2;
    e->head_y = e->height / 2;
    e->dir_x = 0;
    e->dir_y = -1;
    e->tail_len = 3;
    e->tail_x[0] = e->head_x;
    e->tail_y[0] = e->head_y + 1;
    e->tail_x[1] = e->head_x;
    e->tail_y[1] = e->head_y + 2;
    e->tail_x[2] = e->head_x;
    e->tail_y[2] = e->head_y + 3;
    e->prev_score = 0;
    e->score = 0;
    e->steps_alive = 0;
    e->dead = false;

    do {
        e->apple_x = rand_int(0, e->width);
        e->apple_y = rand_int(0, e->height);
    } while (cell_occupied(e, e->apple_x, e->apple_y, true));
}

void snake_env_step(snake_env* e, int action) {
    if (!e || e->dead) return;

    // action: 0=up, 1=right, 2=down, 3=left
    int ndx = 0, ndy = 0;
    switch (action) {
        case 0: ndx = 0;  ndy = -1; break;
        case 1: ndx = 1;  ndy = 0;  break;
        case 2: ndx = 0;  ndy = 1;  break;
        case 3: ndx = -1; ndy = 0;  break;
    }

    // prevent 180 turn
    if (ndx == -e->dir_x && ndy == -e->dir_y) {
        ndx = e->dir_x;
        ndy = e->dir_y;
    }

    e->dir_x = ndx;
    e->dir_y = ndy;

    // move head
    e->head_x += e->dir_x;
    e->head_y += e->dir_y;

    // wall collision = death
    if (e->head_x < 0 || e->head_x >= e->width || e->head_y < 0 || e->head_y >= e->height) {
        e->dead = true;
        return;
    }

    // self collision
    for (int i = 0; i < e->tail_len; ++i) {
        if (e->head_x == e->tail_x[i] && e->head_y == e->tail_y[i]) {
            e->dead = true;
            return;
        }
    }

    // apple check
    bool ate = (e->head_x == e->apple_x && e->head_y == e->apple_y);
    if (ate) {
        e->prev_score = e->score;
        e->score++;
        // grow tail
        for (int i = e->tail_len; i > 0; --i) {
            e->tail_x[i] = e->tail_x[i - 1];
            e->tail_y[i] = e->tail_y[i - 1];
        }
        e->tail_x[0] = e->head_x;
        e->tail_y[0] = e->head_y;
        e->tail_len++;

        // new apple
        do {
            e->apple_x = rand_int(0, e->width);
            e->apple_y = rand_int(0, e->height);
        } while (cell_occupied(e, e->apple_x, e->apple_y, true));
    } else {
        // advance tail
        for (int i = e->tail_len - 1; i > 0; --i) {
            e->tail_x[i] = e->tail_x[i - 1];
            e->tail_y[i] = e->tail_y[i - 1];
        }
        e->tail_x[0] = e->head_x;
        e->tail_y[0] = e->head_y;
    }

    e->steps_alive++;
}

bool snake_env_is_done(const snake_env* e) {
    if (!e) return true;
    return e->dead || e->steps_alive >= e->max_steps;
}

int snake_env_get_observation_size(void) {
    return 9;
}

void snake_env_get_observation(const snake_env* e, float* out) {
    if (!e || !out) return;

    // obs[0]: angle to apple, normalized -1..1
    float dx = (float)(e->apple_x - e->head_x);
    float dy = (float)(e->apple_y - e->head_y);
    float angle = std::atan2(dy, dx);
    out[0] = angle / (float)M_PI;

    // obs[1..4]: clear distance to obstacle in cardinal directions, normalized 0..1
    int du = 0; for (int y = e->head_y - 1; y >= 0 && !cell_occupied(e, e->head_x, y, true); --y) du++;
    int dr = 0; for (int x = e->head_x + 1; x < e->width && !cell_occupied(e, x, e->head_y, true); ++x) dr++;
    int dd = 0; for (int y = e->head_y + 1; y < e->height && !cell_occupied(e, e->head_x, y, true); ++y) dd++;
    int dl = 0; for (int x = e->head_x - 1; x >= 0 && !cell_occupied(e, x, e->head_y, true); --x) dl++;
    out[1] = (float)du / (float)e->height;
    out[2] = (float)dr / (float)e->width;
    out[3] = (float)dd / (float)e->height;
    out[4] = (float)dl / (float)e->width;

    // obs[5..8]: clear distance in diagonal directions
    int dur = 0; for (int d = 1; e->head_x+d<e->width && e->head_y-d>=0 && !cell_occupied(e, e->head_x+d, e->head_y-d, true); ++d) dur++;
    int ddr = 0; for (int d = 1; e->head_x+d<e->width && e->head_y+d<e->height && !cell_occupied(e, e->head_x+d, e->head_y+d, true); ++d) ddr++;
    int ddl = 0; for (int d = 1; e->head_x-d>=0 && e->head_y+d<e->height && !cell_occupied(e, e->head_x-d, e->head_y+d, true); ++d) ddl++;
    int dul = 0; for (int d = 1; e->head_x-d>=0 && e->head_y-d>=0 && !cell_occupied(e, e->head_x-d, e->head_y-d, true); ++d) dul++;
    float max_d = (float)std::max(e->width, e->height);
    out[5] = (float)dur / max_d;
    out[6] = (float)ddr / max_d;
    out[7] = (float)ddl / max_d;
    out[8] = (float)dul / max_d;
}

float snake_env_get_reward(const snake_env* e) {
    if (!e) return 0.0f;
    if (e->dead) return -1.0f;
    // +1 reward on the step where apple was eaten
    if (e->score > e->prev_score) return +1.0f;
    return 0.0f;
}

} // namespace snn
