#include "tournament_recorder.h"
#include "tournament.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdarg>

namespace snn {

static const char* ansi_color(int r, int g, int b) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "\x1b[38;2;%d;%d;%dm", r, g, b);
    return buf;
}

static const char* ansi_reset() { return "\x1b[0m"; }
static const char* ansi_bold() { return "\x1b[1m"; }

static void logf(tournament_recorder* rec, const char* fmt, ...) {
    if (!rec || !rec->log_file) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(rec->log_file, fmt, args);
    va_end(args);
    fprintf(rec->log_file, "\n");
    fflush(rec->log_file);
}

static void jsonf(tournament_recorder* rec, const char* fmt, ...) {
    if (!rec || !rec->json_file) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(rec->json_file, fmt, args);
    va_end(args);
}

tournament_recorder* tournament_recorder_create(const char* log_path, const char* json_path) {
    tournament_recorder* rec = new tournament_recorder;
    rec->log_file = fopen(log_path, "w");
    rec->json_file = fopen(json_path, "w");
    rec->enabled = (rec->log_file != nullptr || rec->json_file != nullptr);
    rec->frame_counter = 0;

    time_t now = time(nullptr);
    if (rec->log_file) {
        fprintf(rec->log_file, "%s=== SNN Tournament Log ===%s\n", ansi_bold(), ansi_reset());
        fprintf(rec->log_file, "Started: %s", ctime(&now));
        fprintf(rec->log_file, "\n");
        fflush(rec->log_file);
    }
    if (rec->json_file) {
        jsonf(rec, "{\n");
        jsonf(rec, "  \"tournament\": {\n");
        jsonf(rec, "    \"started\": \"%s\"\n", ctime(&now));
        jsonf(rec, "    \"generations\": [\n");
        fflush(rec->json_file);
    }
    return rec;
}

void tournament_recorder_destroy(tournament_recorder* rec) {
    if (!rec) return;
    if (rec->json_file) {
        jsonf(rec, "    ]\n");
        jsonf(rec, "  }\n");
        jsonf(rec, "}\n");
        fclose(rec->json_file);
    }
    if (rec->log_file) {
        logf(rec, "%s=== Tournament Complete ===%s", ansi_bold(), ansi_reset());
        fclose(rec->log_file);
    }
    delete rec;
}

void tournament_recorder_begin_generation(tournament_recorder* rec,
                                          tournament_runner* tr,
                                          int generation) {
    if (!rec || !rec->enabled) return;
    rec->frame_counter = 0;

    const char* col = ansi_color(100, 200, 255);
    if (rec->log_file) {
        fprintf(rec->log_file, "\n%s%s--- Generation %d ---%s\n",
                col, ansi_bold(), generation, ansi_reset());
        fprintf(rec->log_file, "Species: %d | Children/species: %d | Max steps: %d\n\n",
                tr->P, tr->K, tr->sim_steps);
        fflush(rec->log_file);
    }
    if (rec->json_file) {
        jsonf(rec, "      {\n");
        jsonf(rec, "        \"generation\": %d,\n", generation);
        jsonf(rec, "        \"species\": [\n");
        fflush(rec->json_file);
    }
}

void tournament_recorder_species_result(tournament_recorder* rec,
                                        tournament_runner* tr,
                                        int species_idx,
                                        const float* fitness) {
    if (!rec || !rec->enabled) return;
    auto& sp = tr->species[species_idx];

    if (rec->log_file) {
        fprintf(rec->log_file, "%sSpecies %d:%s\n", ansi_color(255, 200, 100), species_idx, ansi_reset());
        fprintf(rec->log_file, "  CPPN: nodes=%d edges=%d\n",
                sp.cppn->num_nodes, sp.cppn->num_edges);
        for (int k = 0; k < sp.K; ++k) {
            const char* status = sp.fitness[k] > 0 ? "ALIVE" : "DEAD";
            int cr = sp.fitness[k] > 0 ? 100 : 255;
            int cg = sp.fitness[k] > 0 ? 255 : 100;
            fprintf(rec->log_file, "  %sChild %2d:%s score=%-3d %s\n",
                    ansi_color(cr, cg, 100), k, ansi_reset(),
                    (int)sp.fitness[k], status);
        }
        fflush(rec->log_file);
    }

    if (rec->json_file) {
        jsonf(rec, "          {\n");
        jsonf(rec, "            \"species_idx\": %d,\n", species_idx);
        jsonf(rec, "            \"cppn\": {\"nodes\": %d, \"edges\": %d, \"activations\": [",
              sp.cppn->num_nodes, sp.cppn->num_edges);
        int act_count = 0;
        for (int n = 0; n < sp.cppn->num_nodes && n < 8; ++n) {
            if (n > 0) jsonf(rec, ",");
            const char* a = "L";
            switch (sp.cppn->nodes[n].a) {
                case cppn_activation_function::LINEAR:  a = "L"; break;
                case cppn_activation_function::SIGMOID: a = "S"; break;
                case cppn_activation_function::SIN:     a = "sin"; break;
                case cppn_activation_function::GAUSSIAN:a = "G"; break;
                case cppn_activation_function::ABS:     a = "A"; break;
                case cppn_activation_function::NEG:     a = "N"; break;
                case cppn_activation_function::RELU:    a = "R"; break;
                case cppn_activation_function::TANH:    a = "T"; break;
            }
            jsonf(rec, "\"%s\"", a);
            act_count++;
        }
        if (act_count < sp.cppn->num_nodes) jsonf(rec, ",\"...(%d more)\"", sp.cppn->num_nodes - act_count);
        jsonf(rec, "]},\n");

        jsonf(rec, "            \"children\": [\n");
        for (int k = 0; k < sp.K; ++k) {
            jsonf(rec, "              {\"id\": %d, \"score\": %.0f, \"alive\": %s}%s\n",
                  k, sp.fitness[k], sp.fitness[k] > 0 ? "true" : "false",
                  k < sp.K - 1 ? "," : "");
        }
        jsonf(rec, "            ]\n");
        jsonf(rec, "          }%s\n", species_idx < tr->P - 1 ? "," : "");
        fflush(rec->json_file);
    }
}

void tournament_recorder_end_generation(tournament_recorder* rec,
                                        tournament_runner* tr,
                                        const float* fitness) {
    if (!rec || !rec->enabled) return;

    float best = 0.0f, avg = 0.0f;
    for (int i = 0; i < tr->P * tr->K; ++i) {
        if (fitness[i] > best) best = fitness[i];
        avg += fitness[i];
    }
    avg /= tr->P * tr->K;

    if (rec->log_file) {
        fprintf(rec->log_file, "\n%s=== Generation Complete ===%s\n", ansi_bold(), ansi_reset());
        fprintf(rec->log_file, "Best: %.1f | Avg: %.1f\n\n", best, avg);
        fflush(rec->log_file);
    }
}

void tournament_recorder_child_frame(tournament_recorder* rec,
                                     int species_idx,
                                     int child_idx,
                                     int step,
                                     const snake_env* env,
                                     const bool* motor_spikes,
                                     int motor_spike_count) {
    if (!rec || !rec->enabled || !env) return;
    rec->frame_counter++;

    if (rec->log_file) {
        const char* status = env->dead ? "x" : " ";
        fprintf(rec->log_file, "  [Gen ?] Sp %d Ch %2d Step %4d Score %d %s\r",
                species_idx, child_idx, step, env->score, status);
        fflush(rec->log_file);
    }

    if (rec->json_file) {
        jsonf(rec, "            {\"step\":%d,\"species\":%d,\"child\":%d,"
                    "\"head_x\":%d,\"head_y\":%d,\"dir_x\":%d,\"dir_y\":%d,"
                    "\"tail\":[",
              step, species_idx, child_idx,
              env->head_x, env->head_y, env->dir_x, env->dir_y);
        for (int i = 0; i < env->tail_len; ++i) {
            if (i > 0) jsonf(rec, ",");
            jsonf(rec, "[%d,%d]", env->tail_x[i], env->tail_y[i]);
        }
        jsonf(rec, "],\"apple_x\":%d,\"apple_y\":%d,"
                    "\"width\":%d,\"height\":%d,"
                    "\"motor_spikes\":[",
              env->apple_x, env->apple_y, env->width, env->height);
        for (int m = 0; m < motor_spike_count && m < 4; ++m) {
            if (m > 0) jsonf(rec, ",");
            jsonf(rec, "%d", motor_spikes[m] ? 1 : 0);
        }
        jsonf(rec, "],\"score\":%d,\"alive\":%s},\n",
              env->score, env->dead ? "false" : "true");
        fflush(rec->json_file);
    }
}

void tournament_recorder_cppn_info(tournament_recorder* rec,
                                   const cppn_genome* genome,
                                   const char* label) {
    if (!rec || !rec->log_file || !genome) return;
    fprintf(rec->log_file, "\n%sCPPN Info: %s%s\n", ansi_bold(), label, ansi_reset());
    fprintf(rec->log_file, "  Nodes: %d | Edges: %d\n", genome->num_nodes, genome->num_edges);
    fprintf(rec->log_file, "  Activations: ");
    for (int i = 0; i < genome->num_nodes && i < 8; ++i) {
        switch (genome->nodes[i].a) {
            case cppn_activation_function::LINEAR:  fprintf(rec->log_file, "L "); break;
            case cppn_activation_function::SIGMOID: fprintf(rec->log_file, "S "); break;
            case cppn_activation_function::SIN:     fprintf(rec->log_file, "sin "); break;
            case cppn_activation_function::GAUSSIAN:fprintf(rec->log_file, "G "); break;
            case cppn_activation_function::ABS:     fprintf(rec->log_file, "A "); break;
            case cppn_activation_function::NEG:     fprintf(rec->log_file, "N "); break;
            case cppn_activation_function::RELU:    fprintf(rec->log_file, "R "); break;
            case cppn_activation_function::TANH:    fprintf(rec->log_file, "T "); break;
        }
    }
    fprintf(rec->log_file, "\n\n");
    fflush(rec->log_file);
}

} // namespace snn
