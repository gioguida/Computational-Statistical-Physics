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

struct AnnealingRecord {
    int step;
    double T;
    double H_mean;
    double H_var;
    double C_v;
    double acceptance_rate;
    double H_min_so_far;
    double n_active_mean;
    double n_active_std;
    double delta_E_mean_neg;
    double delta_E_mean_pos;
};

struct AnnealingStateCheckpoint {
    int step;
    double temperature;
    double energy;
    std::vector<int> state;
};

struct AnnealingResult {
    std::vector<int> state;
    std::vector<int> best_state;
    double best_energy = 0.0;
    std::vector<AnnealingTraceSample> trace;
    std::vector<AnnealingRecord> annealing_trace;
    std::vector<AnnealingStateCheckpoint> checkpoints;
};

AnnealingResult main_simulation(int N, interaction_mat_t J, std::vector<double> h,
    double T_min, double T_max, int N_steps, double toll,
    int N_sweeps, int seed, int log_every_steps, int checkpoint_every_steps);

#endif
