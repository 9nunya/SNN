#ifndef SNN_TOURNAMENT_RECORDER_H
#define SNN_TOURNAMENT_RECORDER_H

#include "snake_env.h"
#include <cstdio>

struct tournament_runner;
struct cppn_genome;

namespace snn {

// Tournament data recorder: exports per-generation data to files
// for visualization and debug output.
//
// Outputs:
//   tournament_log.txt  - human-readable ANSI-colored debug output
//   tournament_data.json - machine-readable JSON for video generation

struct tournament_recorder {
    FILE* log_file;
    FILE* json_file;
    bool enabled;
    int frame_counter;
};

tournament_recorder* tournament_recorder_create(const char* log_path, const char* json_path);
void tournament_recorder_destroy(tournament_recorder* rec);

// Write generation start header
void tournament_recorder_begin_generation(tournament_recorder* rec,
                                          tournament_runner* tr,
                                          int generation);

void tournament_recorder_species_result(tournament_recorder* rec,
                                        tournament_runner* tr,
                                        int species_idx,
                                        const float* fitness);

void tournament_recorder_end_generation(tournament_recorder* rec,
                                        tournament_runner* tr,
                                        const float* fitness);

// Write generation summary after evolution
void tournament_recorder_end_generation(tournament_recorder* rec,
                                        const tournament_runner* tr,
                                        const float* fitness);

// Write a single child's game frame (called every N steps from snake_brain)
void tournament_recorder_child_frame(tournament_recorder* rec,
                                     int species_idx,
                                     int child_idx,
                                     int step,
                                     const snake_env* env,
                                     const bool* motor_spikes,
                                     int motor_spike_count);

// Write CPPN genome info for display
void tournament_recorder_cppn_info(tournament_recorder* rec,
                                   const cppn_genome* genome,
                                   const char* label);

}

#endif
