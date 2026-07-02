#include "cppn.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace snn {

// forward declaration helpers
static float apply_act(float x, cppn_activation_function act) {
    switch (act) {
        case cppn_activation_function::LINEAR:   return x;
        case cppn_activation_function::SIGMOID:  return 1.0f / (1.0f + std::exp(-x));
        case cppn_activation_function::SIN:      return std::sin(x);
        case cppn_activation_function::GAUSSIAN: return std::exp(-x * x);
        case cppn_activation_function::ABS:      return std::fabs(x);
        case cppn_activation_function::NEG:      return -x;
        case cppn_activation_function::RELU:     return x > 0.0f ? x : std::exp(x) - 1.0f;
        case cppn_activation_function::TANH:     return std::tanh(x);
    }
    return x;
}

void cppn_forward(const cppn_genome& g, float x1, float y1, float z1,
                  float x2, float y2, float z2, float out[2]) {
    // allocate vals for all nodes
    int n = g.num_nodes;
    if (n == 0) { out[0] = 0.0f; out[1] = 0.0f; return; }

    float* vals = new float[n];
    std::fill(vals, vals + n, 0.0f);

    // set inputs: first 6 nodes are (x1,y1,z1,x2,y2,z2)
    if (n > 0) vals[0] = x1;
    if (n > 1) vals[1] = y1;
    if (n > 2) vals[2] = z1;
    if (n > 3) vals[3] = x2;
    if (n > 4) vals[4] = y2;
    if (n > 5) vals[5] = z2;

    // accumulate: do n+1 passes to ensure signal flows through longest path
    for (int iter = 0; iter < n + 1; ++iter) {
        for (int e = 0; e < g.num_edges; ++e) {
            const cppn_genome_edge& ed = g.edges[e];
            if (!ed.enabled) continue;
            if (ed.from >= 0 && ed.from < n && ed.to >= 0 && ed.to < n) {
                vals[ed.to] += vals[ed.from] * ed.weight;
            }
        }
    }

    // apply activation per node
    for (int i = 0; i < n; ++i) {
        vals[i] = apply_act(vals[i], g.nodes[i].a);
    }

    // outputs: last 2 nodes
    out[0] = (n >= 2) ? vals[n - 2] : 0.0f; // link probability
    out[1] = (n >= 1) ? vals[n - 1] : 0.0f; // weight
    delete[] vals;
}

void cppn_forward_single(const cppn_genome& g, float x, float y, float z, float out[5]) {
    // single-point query: first 3 nodes are (x,y,z), outputs are last 5 nodes
    int n = g.num_nodes;
    if (n == 0) {
        for (int i = 0; i < 5; ++i) out[i] = 0.0f;
        return;
    }

    float* vals = new float[n];
    std::fill(vals, vals + n, 0.0f);

    if (n > 0) vals[0] = x;
    if (n > 1) vals[1] = y;
    if (n > 2) vals[2] = z;

    for (int iter = 0; iter < n + 1; ++iter) {
        for (int e = 0; e < g.num_edges; ++e) {
            const cppn_genome_edge& ed = g.edges[e];
            if (!ed.enabled) continue;
            if (ed.from >= 0 && ed.from < n && ed.to >= 0 && ed.to < n) {
                vals[ed.to] += vals[ed.from] * ed.weight;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        vals[i] = apply_act(vals[i], g.nodes[i].a);
    }

    // last 5 nodes: v_th_scale, tau_rc_scale, tau_ref_scale, max_rate_scale, intercept
    for (int i = 0; i < 5; ++i) {
        int idx = n - 5 + i;
        out[i] = (idx >= 0) ? vals[idx] : 0.0f;
    }
    delete[] vals;
}

// ---- genome lifecycle ----

cppn_genome* cppn_genome_create(void) {
    cppn_genome* g = new cppn_genome;
    g->nodes = nullptr;
    g->edges = nullptr;
    g->num_nodes = 0;
    g->num_edges = 0;
    g->fitness = 0.0f;
    return g;
}

cppn_genome* cppn_genome_clone(const cppn_genome* src) {
    cppn_genome* g = cppn_genome_create();
    g->num_nodes = src->num_nodes;
    g->num_edges = src->num_edges;
    g->fitness = src->fitness;

    if (g->num_nodes > 0) {
        g->nodes = new cppn_genome_node[g->num_nodes];
        std::memcpy(g->nodes, src->nodes, g->num_nodes * sizeof(cppn_genome_node));
    }
    if (g->num_edges > 0) {
        g->edges = new cppn_genome_edge[g->num_edges];
        std::memcpy(g->edges, src->edges, g->num_edges * sizeof(cppn_genome_edge));
    }
    return g;
}

void cppn_genome_destroy(cppn_genome* g) {
    if (!g) return;
    delete[] g->nodes;
    delete[] g->edges;
    delete g;
}

// ---- innovation tracker ----

cppn_innovation_tracker* cppn_innovation_create(void) {
    cppn_innovation_tracker* db = new cppn_innovation_tracker;
    db->next_innovation_ = 0;
    return db;
}

void cppn_innovation_destroy(cppn_innovation_tracker* db) {
    if (db) delete db;
}

void cppn_innovation_reset(cppn_innovation_tracker* db) {
    if (!db) return;
    db->map_.clear();
    db->next_innovation_ = 0;
}

int cppn_innovation_get_or_assign(cppn_innovation_tracker* db, int from, int to) {
    if (!db) return -1;
    auto key = std::make_pair(from, to);
    auto it = db->map_.find(key);
    if (it != db->map_.end()) return it->second;
    int inv = db->next_innovation_++;
    db->map_[key] = inv;
    return inv;
}

// ---- mutation ----

static float randf(float lo, float hi) {
    return lo + (float)std::rand() / RAND_MAX * (hi - lo);
}

static bool has_path(const cppn_genome_edge* edges, int num_edges, int from, int to, int n) {
    // BFS: does a path exist from `from` to `to` using enabled edges?
    if (from == to) return true;
    bool* visited = new bool[n];
    std::fill(visited, visited + n, false);
    int* queue = new int[n];
    int qhead = 0, qtail = 0;
    queue[qtail++] = from;
    visited[from] = true;
    bool found = false;
    while (qhead < qtail && !found) {
        int u = queue[qhead++];
        for (int e = 0; e < num_edges; ++e) {
            const cppn_genome_edge& ed = edges[e];
            if (!ed.enabled) continue;
            if (ed.from == u && !visited[ed.to]) {
                visited[ed.to] = true;
                if (ed.to == to) { found = true; break; }
                queue[qtail++] = ed.to;
            }
        }
    }
    delete[] visited;
    delete[] queue;
    return found;
}

void cppn_mutate(cppn_genome* g, float weight_rate, float weight_std,
                 float add_node_rate, float add_conn_rate,
                 cppn_innovation_tracker* db) {
    if (!g || g->num_nodes == 0) return;

    // 1. weight mutation: every enabled edge has a chance to be perturbed
    for (int e = 0; e < g->num_edges; ++e) {
        cppn_genome_edge& ed = g->edges[e];
        if (!ed.enabled) continue;
        if ((float)std::rand() / RAND_MAX < weight_rate) {
            ed.weight += randf(-weight_std, weight_std);
            if (ed.weight > 3.0f) ed.weight = 3.0f;
            if (ed.weight < -3.0f) ed.weight = -3.0f;
        }
    }

    // 2. add node mutation: split a random enabled edge into two edges via a new hidden node
    if ((float)std::rand() / RAND_MAX < add_node_rate && g->num_edges > 0) {
        // collect enabled edges
        int* enabled_idxs = new int[g->num_edges];
        int n_enabled = 0;
        for (int e = 0; e < g->num_edges; ++e) {
            if (g->edges[e].enabled) enabled_idxs[n_enabled++] = e;
        }
        if (n_enabled > 0) {
            int pick = enabled_idxs[std::rand() % n_enabled];
            cppn_genome_edge& ed = g->edges[pick];
            ed.enabled = false;

            int new_id = g->num_nodes;
            int old_to = ed.to;
            int old_from = ed.from;
            float old_w = ed.weight;

            // append new node
            cppn_genome_node* new_nodes = new cppn_genome_node[g->num_nodes + 1];
            std::memcpy(new_nodes, g->nodes, g->num_nodes * sizeof(cppn_genome_node));
            new_nodes[g->num_nodes] = { new_id, cppn_activation_function::SIGMOID };
            delete[] g->nodes;
            g->nodes = new_nodes;
            g->num_nodes++;

            // append two new edges: from->new_id (w=1), new_id->to (w=old_w)
            int inv1 = cppn_innovation_get_or_assign(db, old_from, new_id);
            int inv2 = cppn_innovation_get_or_assign(db, new_id, old_to);

            cppn_genome_edge* new_edges = new cppn_genome_edge[g->num_edges + 2];
            std::memcpy(new_edges, g->edges, g->num_edges * sizeof(cppn_genome_edge));
            new_edges[g->num_edges]     = { inv1, old_from, new_id, 1.0f, true };
            new_edges[g->num_edges + 1] = { inv2, new_id, old_to, old_w, true };
            delete[] g->edges;
            g->edges = new_edges;
            g->num_edges += 2;
        }
        delete[] enabled_idxs;
    }

    // 3. add connection mutation: new edge between two unconnected nodes
    if ((float)std::rand() / RAND_MAX < add_conn_rate && g->num_nodes >= 2) {
        // find all valid (from, to) pairs without existing enabled edge and no cycle
        int max_pairs = g->num_nodes * (g->num_nodes - 1);
        int* from_buf = new int[max_pairs];
        int* to_buf = new int[max_pairs];
        int n_pairs = 0;

        for (int a = 0; a < g->num_nodes; ++a) {
            for (int b = 0; b < g->num_nodes; ++b) {
                if (a == b) continue;
                // check if edge a->b already exists
                bool exists = false;
                for (int e = 0; e < g->num_edges; ++e) {
                    if (g->edges[e].enabled && g->edges[e].from == a && g->edges[e].to == b) {
                        exists = true; break;
                    }
                }
                if (exists) continue;
                // avoid adding edge that would create a direct cycle b->a already exists
                bool reverse = false;
                for (int e = 0; e < g->num_edges; ++e) {
                    if (g->edges[e].enabled && g->edges[e].from == b && g->edges[e].to == a) {
                        reverse = true; break;
                    }
                }
                if (reverse) continue;
                from_buf[n_pairs] = a;
                to_buf[n_pairs] = b;
                n_pairs++;
            }
        }

        if (n_pairs > 0) {
            int pick = std::rand() % n_pairs;
            int a = from_buf[pick], b = to_buf[pick];
            int inv = cppn_innovation_get_or_assign(db, a, b);
            float w = randf(-1.0f, 1.0f);

            cppn_genome_edge* new_edges = new cppn_genome_edge[g->num_edges + 1];
            std::memcpy(new_edges, g->edges, g->num_edges * sizeof(cppn_genome_edge));
            new_edges[g->num_edges] = { inv, a, b, w, true };
            delete[] g->edges;
            g->edges = new_edges;
            g->num_edges++;
        }
        delete[] from_buf;
        delete[] to_buf;
    }
}

// ---- crossover ----

cppn_genome cppn_crossover(const cppn_genome* p1, const cppn_genome* p2) {
    // assume p1 is fitter or equal; if p2 fitter, swap
    const cppn_genome* better = &p1[0];
    const cppn_genome* worse  = &p2[0];
    // fitness comparison done outside; just use p1 as better by convention
    // caller should order parents

    cppn_genome child;
    child.fitness = 0.0f;

    // nodes: union
    int max_nodes = std::max(p1->num_nodes, p2->num_nodes);
    child.num_nodes = max_nodes;
    child.nodes = new cppn_genome_node[max_nodes];
    std::memcpy(child.nodes, better->nodes, better->num_nodes * sizeof(cppn_genome_node));
    if (worse->num_nodes > better->num_nodes) {
        std::memcpy(child.nodes + better->num_nodes,
                    worse->nodes + better->num_nodes,
                    (worse->num_nodes - better->num_nodes) * sizeof(cppn_genome_node));
    }

    // edges: aligned by innovation number
    // build lookup for worse parent
    std::unordered_map<int, cppn_genome_edge> worse_map;
    for (int e = 0; e < worse->num_edges; ++e) {
        worse_map[worse->edges[e].innovation] = worse->edges[e];
    }

    child.num_edges = 0;
    child.edges = nullptr;

    // count matching + disjoint from better
    std::vector<cppn_genome_edge> matched;
    std::vector<cppn_genome_edge> disjoint;

    for (int e = 0; e < better->num_edges; ++e) {
        const cppn_genome_edge& be = better->edges[e];
        auto it = worse_map.find(be.innovation);
        if (it != worse_map.end()) {
            // matching gene: randomly pick from either parent
            if ((float)std::rand() / RAND_MAX < 0.5f) {
                matched.push_back(be);
            } else {
                matched.push_back(it->second);
            }
        } else {
            // disjoint/excess: inherit from fitter parent
            disjoint.push_back(be);
        }
    }

    child.num_edges = matched.size() + disjoint.size();
    child.edges = new cppn_genome_edge[child.num_edges];
    int idx = 0;
    for (size_t i = 0; i < matched.size(); ++i) child.edges[idx++] = matched[i];
    for (size_t i = 0; i < disjoint.size(); ++i) child.edges[idx++] = disjoint[i];

    // small chance to disable random gene in child
    for (int e = 0; e < child.num_edges; ++e) {
        if ((float)std::rand() / RAND_MAX < 0.05f) {
            child.edges[e].enabled = false;
        }
    }

    return child;
}

// ---- distance for speciation ----

float cppn_distance(const cppn_genome* a, const cppn_genome* b,
                    float c1, float c2, float c3) {
    if (!a || !b || a->num_nodes == 0 || b->num_nodes == 0) return 0.0f;

    // build innovation lookups
    std::unordered_map<int, const cppn_genome_edge*> a_map;
    for (int e = 0; e < a->num_edges; ++e) {
        if (a->edges[e].enabled) a_map[a->edges[e].innovation] = &a->edges[e];
    }
    std::unordered_map<int, const cppn_genome_edge*> b_map;
    for (int e = 0; e < b->num_edges; ++e) {
        if (b->edges[e].enabled) b_map[b->edges[e].innovation] = &b->edges[e];
    }

    int max_innov = 0;
    for (int e = 0; e < a->num_edges; ++e)
        if (a->edges[e].innovation > max_innov) max_innov = a->edges[e].innovation;
    for (int e = 0; e < b->num_edges; ++e)
        if (b->edges[e].innovation > max_innov) max_innov = b->edges[e].innovation;
    if (max_innov == 0) return 0.0f;

    int matching = 0, disjoint = 0, excess = 0;
    float weight_diff = 0.0f;

    int max_a = 0, max_b = 0;
    for (int e = 0; e < a->num_edges; ++e)
        if (a->edges[e].innovation > max_a) max_a = a->edges[e].innovation;
    for (int e = 0; e < b->num_edges; ++e)
        if (b->edges[e].innovation > max_b) max_b = b->edges[e].innovation;

    int max_edge_id = std::max(max_a, max_b);

    for (int inv = 0; inv <= max_edge_id; ++inv) {
        bool in_a = a_map.count(inv) > 0;
        bool in_b = b_map.count(inv) > 0;
        if (in_a && in_b) {
            matching++;
            weight_diff += std::fabs(a_map[inv]->weight - b_map[inv]->weight);
        } else if (in_a || in_b) {
            if (inv <= std::min(max_a, max_b))
                disjoint++;
            else
                excess++;
        }
    }

    float n = (float)std::max(a->num_edges, b->num_edges);
    if (n < 1.0f) n = 1.0f;

    float avg_wd = matching > 0 ? weight_diff / matching : 0.0f;
    return (c1 * excess + c2 * disjoint + c3 * avg_wd) / n;
}

} // namespace snn
