#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "collections.h"
#include "simulation.h"

int main() {
    int NUM_NEURONS = 1;
    snn::adaptive_neuron_creation_parameters<float> n_params;
    n_params.v_th = 1.0f;
    n_params.tau_rc = 0.02f;
    n_params.tau_ref = 0.002f;
    n_params.v_init = 0.0f;
    n_params.slf = true;
    n_params.tau_w = 0.3f;
    n_params.b = 0.2f;

    snn::range<float> max_rate_range = {200.0f, 400.0f};
    snn::range<float> intercept_range = {-1.0f, 1.0f};
    std::vector<float> encoders = {1.0f};

    auto* lifs = snn::neuron_collection_init<float>(NUM_NEURONS, n_params, max_rate_range, intercept_range, encoders);
    snn::conductance_synapse_creation_parameters<float> s_params;
    s_params.tau_s = 0.01f;
    s_params.E_rev = 0.0f;
    auto* synapses = snn::synapse_collection_init<float>(NUM_NEURONS, s_params);

    snn::simulation_creation_params<float> sc_params;
    sc_params.backend = snn::backend_kind::CUDA;
    sc_params.T_step = 0.001f;
    auto* sim = snn::simulation_init<float>(sc_params);
    sim->s->neurons = lifs;
    sim->s->synapses = synapses;

    float T_max = 8.0f;
    int steps = static_cast<int>(T_max / sc_params.T_step);
    sim->s->inputs = new float[NUM_NEURONS * steps];
    for (int step = 0; step < steps; ++step) {
        float t = step * sc_params.T_step;
        float I = 0.0f;
        if (t >= 2.0f && t < 3.0f) I = -0.5f;
        else if (t >= 5.0f) I = 0.3f;
        sim->s->inputs[step * NUM_NEURONS] = I;
    }

    simulation_run_for(sim, T_max);

    int n_stride = sim->s->neuron_stride;
    int s_stride = sim->s->synapse_stride;
    int total_stride = n_stride + s_stride;

    std::vector<float> v_log(steps);
    std::vector<float> v_th_log(steps);
    std::vector<float> out_log(steps);
    std::vector<float> in_log(steps);
    std::vector<float> synapse_log(steps);
    std::vector<float> w_log(steps);

    for (int i = 0; i < steps; ++i) {
        in_log[i] = sim->s->inputs[i * NUM_NEURONS];
        auto *state = lifs->neurons[0]->s;
        float effective_v_th;
        float w_val = 0.0f;
        if (state->type == snn::neuron_type::ALIF) {
            auto *alif = static_cast<snn::adaptive_neuron_state<float>*>(state);
            effective_v_th = state->v_th + alif->w;
            w_val = alif->w;
        } else {
            effective_v_th = state->v_th;
        }
        v_th_log[i] = effective_v_th;
        out_log[i] = sim->s->outputs[i * total_stride];
        v_log[i] = state->v;
        synapse_log[i] = sim->s->outputs[i * total_stride + n_stride];
        w_log[i] = w_val;
    }

    std::ofstream out("output.csv");
    out << "step,input,v,v_th_eff,spike,synapse,w\n";
    for (int i = 0; i < steps; ++i) {
        out << i << "," << in_log[i] << "," << v_log[i] << "," << v_th_log[i] << "," << out_log[i] << "," << synapse_log[i] << "," << w_log[i] << "\n";
    }
    out.close();

    std::cout << "Simulation complete. cur_time=" << sim->s->cur_time << std::endl;
    return 0;
}
