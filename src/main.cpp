#include <cstdio>
#include <cstdlib>
#include "brain_builder.h"
#include "tournament.h"
#include "tournament_recorder.h"

int main() {
    // ============================================================
    // Build the brain recipe (cortex layout + wiring rules)
    // ============================================================
    snn::brain_builder* bb = snn::brain_builder_create();

    // Sensory cortex: 9 neurons on the left side
    // Encodes: apple direction + 4 cardinal + 4 diagonal danger rays
    snn::builder_cortex sensory = {
        .region = {{-1.0f, -1.0f, -1.0f}, {-0.6f, 1.0f, 1.0f}},
        .num_neurons = 9,
        .neuron_params = {1.0f, 0.02f, 0.002f, 0.0f, false, snn::neuron_type::ALIF},
        .max_rate_range = {50.0f, 200.0f},
        .intercept_range = {-1.0f, 1.0f},
        .encoder_choices = {1.0f}
    };
    snn::brain_builder_add_cortex(bb, &sensory);

    // Internal field: 200 neurons for recurrent processing
    snn::builder_cortex internal = {
        .region = {{-0.5f, -1.0f, -1.0f}, {0.5f, 1.0f, 1.0f}},
        .num_neurons = 200,
        .neuron_params = {1.0f, 0.02f, 0.002f, 0.0f, false, snn::neuron_type::ALIF},
        .max_rate_range = {50.0f, 200.0f},
        .intercept_range = {-1.0f, 1.0f},
        .encoder_choices = {1.0f}
    };
    snn::brain_builder_add_cortex(bb, &internal);

    // Motor cortex: 4 neurons on the right side (up, right, down, left)
    snn::builder_cortex motor = {
        .region = {{0.6f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},
        .num_neurons = 4,
        .neuron_params = {1.0f, 0.02f, 0.002f, 0.0f, false, snn::neuron_type::ALIF},
        .max_rate_range = {50.0f, 200.0f},
        .intercept_range = {-1.0f, 1.0f},
        .encoder_choices = {1.0f}
    };
    snn::brain_builder_add_cortex(bb, &motor);

    // Wiring rules: sensory -> internal -> motor, plus internal recurrence
    snn::builder_wire_rule r1 = {0, 1, 0.5f, 0.3f};  // sensory -> internal
    snn::brain_builder_add_wire_rule(bb, &r1);

    snn::builder_wire_rule r2 = {1, 2, 0.5f, 0.3f};  // internal -> motor
    snn::brain_builder_add_wire_rule(bb, &r2);

    snn::builder_wire_rule r3 = {1, 1, 0.4f, 0.3f};  // internal recurrent
    snn::brain_builder_add_wire_rule(bb, &r3);

    // ============================================================
    // Create tournament: P species, K children each
    // ============================================================
    const int P = 20;   // number of species
    const int K = 5;    // children per species
    const int sim_steps = 2000;

    snn::tournament_runner* tr = snn::tournament_create(P, K, sim_steps, bb);

    printf("=== SNN Brain Slaughterhouse ===\n");
    printf("Species: %d | Children per species: %d | Steps per game: %d\n", P, K, sim_steps);
    printf("Brain: %d neurons (9 sensory + 200 internal + 4 motor)\n", 9 + 200 + 4);
    printf("\n");

    // ============================================================
    // Run evolution loop
    // ============================================================
    for (int gen = 0; gen < 100; ++gen) {
        if (tr->recorder && tr->recorder->enabled) {
            tournament_recorder_begin_generation(tr->recorder, tr, gen);
        }
        float* fitness = snn::tournament_run(tr);
        snn::tournament_print_stats(tr, fitness);
        snn::tournament_evolve(tr, fitness);
        delete[] fitness;
    }

    // ============================================================
    // Cleanup
    // ============================================================
    snn::tournament_destroy(tr);
    snn::brain_builder_destroy(bb);

    printf("\nSlaughterhouse complete. Best brains evolved.\n");
    return 0;
}
