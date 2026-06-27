#include <iostream>
#include <fstream>
#include <vector>
#include "collections.h"

int main() {
    int NUM_NEURONS = 1;
    snn::neuron_creation_parameters<float> n_params;
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
    auto* synapses = snn::synapse_collection_init<float>(NUM_NEURONS, 0.01f);

    float T_step = 0.001f;
    float T_max = 5.0f;
    int total_steps = static_cast<int>(T_max / T_step);

    // Buffers to store internal state for visualization
    std::vector<float> v_log(total_steps);
    std::vector<float> v_th_log(total_steps);
    std::vector<float> out_log(total_steps);
    std::vector<float> in_log(total_steps, 1.3f);

    // Run simulation manually for single neuron logging
    for(int i=0; i < total_steps; ++i) {
        in_log[i] = 1.3f;
        v_th_log[i] = lifs->neurons[0]->s->v_th + lifs->neurons[0]->s->w;
        out_log[i] = snn::neuron_forward(lifs->neurons[0]->s, in_log[i], T_step);
        v_log[i] = lifs->neurons[0]->s->v;
    }

    std::ofstream out("output.csv");
    out << "step,input,v,v_th_eff,spike\n";
    for(int i=0; i < total_steps; ++i) {
        out << i << "," << in_log[i] << "," << v_log[i] << "," << v_th_log[i] << "," << out_log[i] << "\n";
    }
    out.close();

    std::cout << "Single neuron data saved to output.csv" << std::endl;
    return 0;
}