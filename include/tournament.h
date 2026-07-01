#ifndef SNN_TOURNAMENT_H
#define SNN_TOURNAMENT_H

#include "brain.h"
#include "cppn.h"
#include "snake_brain.h"
#include "brain_builder.h"
#include "tournament_recorder.h"

namespace snn {

// One species in the tournament: one CPPN + K children (snake_brains).
struct tournament_species {
    cppn_genome* cppn;
    snake_brain** brains; // [K]
    float* fitness;       // [K]
    int K;
};

struct tournament_runner {
    tournament_species* species; // [P]
    brain_builder*      builder; // shared brain recipe
    tournament_recorder* recorder; // debug log + JSON export
    int P;
    int K;                       // children per species
    int sim_steps;
    int   generation;
    float species_dist_threshold; // for speciation
};

tournament_runner* tournament_create(int P, int K, int sim_steps, brain_builder* bb);
void              tournament_destroy(tournament_runner* tr);

// Runs all species through their snake games.
// Returns fitness array of size P*K (caller must delete[]).
float* tournament_run(tournament_runner* tr);

// Evolve: speciation via cppn_distance, select top 50%, mutate to refill.
void tournament_evolve(tournament_runner* tr, const float* fitness);

void tournament_print_stats(tournament_runner* tr, const float* fitness);

}

#endif
