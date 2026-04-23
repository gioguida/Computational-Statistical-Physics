#ifndef MAIN_SIMULATION_HPP
#define MAIN_SIMULATION_HPP

#include<iostream>
#include<fstream>
#include<chrono>
#include<filesystem>
#include"spinglass.hpp"

std::vector<int> main_simulation(int N, interaction_mat_t J, std::vector<double> h, 
    double T_min, double T_max, double T_step, double toll,
    int N_sweeps, int seed);

#endif