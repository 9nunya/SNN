#ifndef SNN_CPPN_H
#define SNN_CPPN_H

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace snn {

    enum class cppn_activation_function : uint8_t {
        LINEAR, // f(x) = x
        SIGMOID, // f(x) = 1 / (1 + exp(-x))
        SIN, // f(x) = sin(x)
        GAUSSIAN, // f(x) = exp(-x*x)
        ABS, // f(x) = |x|
        NEG, // f(x) = -x
        RELU, // f(x) = x if x > 0 else exp(x) - 1
        TANH, // f(x) = tanh(x)
    };

    struct cppn_genome_node {
        int id;
        cppn_activation_function a;
    };

    struct cppn_genome_edge {
        int innovation;
        int from;
        int to;
        float weight;
        bool enabled;
    };

    struct cppn_genome {
        cppn_genome_node *nodes;
        cppn_genome_edge *edges;
        int num_nodes;
        int num_edges;
        float fitness;
    };

    struct cppn_innovation_tracker {
        struct pair_hash {
            std::size_t operator()(const std::pair<int, int>& p) const noexcept {
                return (static_cast<std::size_t>(p.first) << 32u)
                    ^ static_cast<std::size_t>(p.second);
            }
        };

        std::unordered_map<std::pair<int, int>, int, pair_hash> map_;
        int next_innovation_;
    };

    // forward pass: pair query, 6 inputs -> 2 outputs
    void cppn_forward(const cppn_genome& g, float x1, float y1, float z1,
                      float x2, float y2, float z2, float out[2]);

    // forward pass: single-point query, 3 inputs -> 5 outputs
    // out[0]=v_th_scale, out[1]=tau_rc_scale, out[2]=tau_ref_scale,
    // out[3]=max_rate_scale, out[4]=intercept
    void cppn_forward_single(const cppn_genome& g, float x, float y, float z, float out[5]);

    // genome lifecycle
    cppn_genome* cppn_genome_create(void);
    cppn_genome* cppn_genome_clone(const cppn_genome* src);
    void cppn_genome_destroy(cppn_genome* g);

    // innovation tracker
    cppn_innovation_tracker* cppn_innovation_create(void);
    void cppn_innovation_destroy(cppn_innovation_tracker* db);
    int  cppn_innovation_get_or_assign(cppn_innovation_tracker* db, int from, int to);
    void cppn_innovation_reset(cppn_innovation_tracker* db);

    // real NEAT operators
    void cppn_mutate(cppn_genome* g, float weight_rate, float weight_std,
                     float add_node_rate, float add_conn_rate,
                     cppn_innovation_tracker* db);
    cppn_genome cppn_crossover(const cppn_genome* p1, const cppn_genome* p2);
    float cppn_distance(const cppn_genome* a, const cppn_genome* b,
                        float c1, float c2, float c3);

}

#endif
