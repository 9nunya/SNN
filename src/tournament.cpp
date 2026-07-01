#include "tournament.h"
#include "tournament_recorder.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

namespace snn {

tournament_runner* tournament_create(int P, int K, int sim_steps, brain_builder* bb) {
    tournament_runner* tr = new tournament_runner;
    tr->P = P;
    tr->K = K;
    tr->sim_steps = sim_steps;
    tr->species = new tournament_species[P];
    tr->generation = 0;
    tr->builder = bb;
    tr->species_dist_threshold = 3.0f;
    tr->recorder = tournament_recorder_create("tournament_log.txt", "tournament_data.json");

    std::srand((unsigned int)std::time(NULL));

    for (int i = 0; i < P; ++i) {
        tr->species[i].cppn = cppn_genome_create();
        tr->species[i].cppn->nodes = new cppn_genome_node[8];
        tr->species[i].cppn->edges = nullptr;
        tr->species[i].cppn->num_nodes = 8;
        tr->species[i].cppn->num_edges = 0;
        for (int n = 0; n < 6; ++n)
            tr->species[i].cppn->nodes[n] = { n, cppn_activation_function::LINEAR };
        tr->species[i].cppn->nodes[6] = { 6, cppn_activation_function::SIGMOID };
        tr->species[i].cppn->nodes[7] = { 7, cppn_activation_function::LINEAR };

        tr->species[i].K = K;
        tr->species[i].brains = new snake_brain*[K];
        tr->species[i].fitness = new float[K];

        for (int k = 0; k < K; ++k) {
            tr->species[i].brains[k] = snake_brain_create(bb, tr->species[i].cppn,
                                                          sim_steps, 0, i,
                                                          tr->recorder);
            if (tr->species[i].brains[k]) {
                tr->species[i].brains[k]->child_idx = k;
            }
            tr->species[i].fitness[k] = 0.0f;
        }
    }

    return tr;
}

float* tournament_run(tournament_runner* tr) {
    float* fitness = new float[tr->P * tr->K];

    for (int i = 0; i < tr->P; ++i) {
        auto& sp = tr->species[i];

        if (tr->recorder && tr->recorder->enabled) {
            tournament_recorder_cppn_info(tr->recorder, sp.cppn, "Parent");
        }

        for (int k = 0; k < sp.K; ++k) {
            if (!sp.brains[k]) {
                fitness[i * tr->K + k] = 0.0f;
                continue;
            }

            snake_brain_reset(sp.brains[k]);
            while (!snake_brain_is_done(sp.brains[k])) {
                snake_brain_step(sp.brains[k]);
            }

            sp.fitness[k] = (float)snake_brain_get_score(sp.brains[k]);
            fitness[i * tr->K + k] = sp.fitness[k];
        }

        if (tr->recorder && tr->recorder->enabled) {
            tournament_recorder_species_result(tr->recorder, tr, i, fitness + i * tr->K);
        }
    }

    return fitness;
}

void tournament_evolve(tournament_runner* tr, const float* fitness) {
    // Compute average fitness per species
    float* species_fitness = new float[tr->P];
    for (int i = 0; i < tr->P; ++i) {
        species_fitness[i] = 0.0f;
        for (int k = 0; k < tr->species[i].K; ++k) {
            species_fitness[i] += fitness[i * tr->K + k];
        }
        species_fitness[i] /= tr->species[i].K;
    }

    // Speciation: group CPPNs by topological distance
    int num_species = 0;
    int* species_assign = new int[tr->P];
    int* species_rep = new int[tr->P];

    for (int i = 0; i < tr->P; ++i) {
        species_assign[i] = -1;
    }

    for (int i = 0; i < tr->P; ++i) {
        bool placed = false;
        for (int s = 0; s < num_species; ++s) {
            float d = cppn_distance(tr->species[i].cppn, tr->species[species_rep[s]].cppn,
                                    1.0f, 1.0f, 0.4f);
            if (d < tr->species_dist_threshold) {
                species_assign[i] = s;
                placed = true;
                break;
            }
        }
        if (!placed) {
            species_assign[i] = num_species;
            species_rep[num_species] = i;
            num_species++;
        }
    }

    // Selection per species: keep top 50%, mutate to refill
    cppn_innovation_tracker* db = cppn_innovation_create();
    cppn_innovation_reset(db);

    for (int s = 0; s < num_species; ++s) {
        // collect members
        int* members = new int[tr->P];
        int n_members = 0;
        for (int i = 0; i < tr->P; ++i) {
            if (species_assign[i] == s) {
                members[n_members++] = i;
            }
        }

        if (n_members == 0) { delete[] members; continue; }

        // sort by fitness descending (simple bubble sort for small groups)
        for (int i = 0; i < n_members; ++i)
            for (int j = i + 1; j < n_members; ++j) {
                if (species_fitness[members[j]] > species_fitness[members[i]]) {
                    int t = members[i]; members[i] = members[j]; members[j] = t;
                }
            }

        int keep = std::max(1, n_members / 2);
        for (int i = 0; i < n_members; ++i) {
            int parent = members[i % keep];
            cppn_genome_destroy(tr->species[i].cppn);
            tr->species[i].cppn = cppn_genome_clone(tr->species[parent].cppn);
            cppn_mutate(tr->species[i].cppn,
                         0.1f, 0.2f, 0.05f, 0.1f, db);
        }

        delete[] members;
    }

    delete[] species_assign;
    delete[] species_rep;
    delete[] species_fitness;
    cppn_innovation_destroy(db);
    tr->generation++;
}

void tournament_print_stats(tournament_runner* tr, const float* fitness) {
    float best = 0.0f, avg = 0.0f;
    int total = tr->P * tr->K;
    for (int i = 0; i < total; ++i) {
        if (fitness[i] > best) best = fitness[i];
        avg += fitness[i];
    }
    avg /= total;

    printf("Gen %d | best=%.1f avg=%.1f\n", tr->generation, best, avg);

    if (tr->recorder && tr->recorder->enabled) {
        tournament_recorder_end_generation(tr->recorder, tr, fitness);
    }
}

void tournament_destroy(tournament_runner* tr) {
    if (!tr) return;
    for (int i = 0; i < tr->P; ++i) {
        for (int k = 0; k < tr->species[i].K; ++k) {
            snake_brain_destroy(tr->species[i].brains[k]);
        }
        delete[] tr->species[i].brains;
        delete[] tr->species[i].fitness;
        cppn_genome_destroy(tr->species[i].cppn);
    }
    delete[] tr->species;
    if (tr->recorder) {
        tournament_recorder_destroy(tr->recorder);
    }
    delete tr;
}

} // namespace snn
