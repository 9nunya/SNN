#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "collections.h"
#include "simulation.h"

void write_csv(const char* filename, snn::simulation<float>* sim, const float* inputs, int steps, int n_neurons, int n_stride, int s_stride, int total_stride) {
    std::ofstream out(filename);
    out << "step,input,v,v_th_eff,spike,synapse,w\n";
    for (int i = 0; i < steps; ++i) {
        float v = sim->s->v_logs ? sim->s->v_logs[i] : 0.0f;
        float v_th = sim->s->v_th_logs ? sim->s->v_th_logs[i] : 0.0f;
        float w = sim->s->w_logs ? sim->s->w_logs[i] : 0.0f;
        float spike = sim->s->outputs ? sim->s->outputs[i * total_stride] : 0.0f;
        float synapse = sim->s->outputs ? sim->s->outputs[i * total_stride + n_stride] : 0.0f;
        float input = inputs[i * n_neurons];
        out << i << "," << input << "," << v << "," << v_th << "," << spike << "," << synapse << "," << w << "\n";
    }
    out.close();
}

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
    auto* synapses = snn::synapse_collection_init<float>(NUM_NEURONS,
        snn::conductance_synapse_creation_parameters<float>{ 0.01f, snn::synapse_type::CONDUCTANCE, 0.0f });

    snn::simulation_creation_params<float> sc_params;
    sc_params.backend = snn::backend_kind::CPU;
    sc_params.T_step = 0.001f;
    auto* sim_cpu = snn::simulation_init<float>(sc_params);
    sim_cpu->s->neurons = lifs;
    sim_cpu->s->synapses = synapses;

    float T_max = 8.0f;
    int steps = static_cast<int>(T_max / sc_params.T_step);
    sim_cpu->s->inputs = new float[NUM_NEURONS * steps];
    for (int step = 0; step < steps; ++step) {
        float t = step * sc_params.T_step;
        float I = 0.0f;
        if (t >= 2.0f && t < 3.0f) I = -0.5f;
        else if (t >= 5.0f) I = 0.3f;
        sim_cpu->s->inputs[step * NUM_NEURONS] = I * 3;
    }

    snn::simulation_run_for(sim_cpu, T_max);
    int n_stride = sim_cpu->s->neuron_stride;
    int s_stride = sim_cpu->s->synapse_stride;
    int total_stride = n_stride + s_stride;

    std::ofstream out_cpu("output_cpu.csv");
    out_cpu << "step,input,v,v_th_eff,spike,synapse,w\n";
    for (int i = 0; i < steps; ++i) {
        float v = sim_cpu->s->v_logs[i];
        float v_th = sim_cpu->s->v_th_logs[i];
        float w = sim_cpu->s->w_logs[i];
        float spike = sim_cpu->s->outputs[i * total_stride + 0];
        float synapse = sim_cpu->s->outputs[i * total_stride + n_stride];
        float input = sim_cpu->s->inputs[i * NUM_NEURONS];
        out_cpu << i << "," << input << "," << v << "," << v_th << "," << spike << "," << synapse << "," << w << "\n";
    }
    out_cpu.close();
    std::cout << "CPU simulation complete. cur_time=" << sim_cpu->s->cur_time << std::endl;

    sc_params.backend = snn::backend_kind::CUDA;
    auto* sim_cuda = snn::simulation_init<float>(sc_params);
    sim_cuda->s->neurons = lifs;
    sim_cuda->s->synapses = synapses;
    sim_cuda->s->inputs = sim_cpu->s->inputs;

    snn::simulation_run_for(sim_cuda, T_max);
    std::ofstream out_cuda("output_cuda.csv");
    out_cuda << "step,input,v,v_th_eff,spike,synapse,w\n";
    for (int i = 0; i < steps; ++i) {
        float v = sim_cuda->s->v_logs[i];
        float v_th = sim_cuda->s->v_th_logs[i];
        float w = sim_cuda->s->w_logs[i];
        float spike = sim_cuda->s->outputs[i * total_stride + 0];
        float synapse = sim_cuda->s->outputs[i * total_stride + n_stride];
        float input = sim_cpu->s->inputs[i * NUM_NEURONS];
        out_cuda << i << "," << input << "," << v << "," << v_th << "," << spike << "," << synapse << "," << w << "\n";
    }
    out_cuda.close();
    std::cout << "CUDA simulation complete. cur_time=" << sim_cuda->s->cur_time << std::endl;

    std::ofstream out_default("output.csv");
    std::ifstream cpu_file("output_cpu.csv");
    out_default << cpu_file.rdbuf();
    out_default.close();
    cpu_file.close();

    return 0;
}