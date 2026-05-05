#ifndef MAIN_SIMULATION_HPP
#define MAIN_SIMULATION_HPP

#include<iostream>
#include<fstream>
#include<chrono>
#include<filesystem>
#include<vector>
#include"spinglass.hpp"

struct AnnealingTraceSample {
    int step;
    double temperature;
    double energy;
    int n_selected;
};

struct AnnealingResult {
    std::vector<int> state;
    std::vector<int> best_state;
    double best_energy = 0.0;
    std::vector<AnnealingTraceSample> trace;
};

AnnealingResult main_simulation(int N, interaction_mat_t J, std::vector<double> h,
    double T_min, double T_max, int N_steps, double toll,
    int N_sweeps, int seed, int log_every_steps);

#endif
