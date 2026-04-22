#include"annealing.hpp"
#include<list>
#include <algorithm>


#define N_energy 5

std::vector<int> main_simulation(int N, interaction_mat_t J, std::vector<double> h, 
    double T_min, double T_max, double T_step, 
    int N_sweeps_eq, int N_sweeps_meas, int seed) {
    // set simulation parameters
    int N_temp = static_cast<int>((T_max - T_min)/T_step);

    std::filesystem::create_directory("../../results");
    
    std::cout << "--- Starting simulation ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Express smulation iterations in sweeps
    int sweep = N;
    int N_eq = N_sweeps_eq * sweep;
    int N_meas = N_sweeps_meas * sweep;
    int meas_int = sweep; 

    std::vector<double> spins(N);

    // instantiate ising model
    Spinglass model(N, J, h, seed);
    model.set_T(T_max); 
    model.compute_initial_energy();

    // equilibrate the initial model at T_max
    std::cout << "--- Equilibrating the initial model ---" << std::endl;
    for(int i = 0; i < 4*N_eq; ++i) {
        model.step();
    }

    std::cout << "--- Beginning Simulation ---" << std::endl;

    std::list<double> energy_list;
    int t = 0;
    double dev = toll + 1;

    while(t < N_temp && toll < dev ) {

        double T = T_max - t*T_step;
        model.set_T(T);

        std::cout << "\n--- Step " << t << " of " << N_temp <<" ---\n";
        std::cout << " Temperature = " << T << std::endl;


        // equilibrate the model
        for(int i = 0; i < N_eq; ++i) {
            model.step();
        }
        
        energy_list.push_back(model.get_energy());
        if (t > N_energy)   energy_list.pop_front();

        double temp = 0;
        for (int i = 0; i < energy_list.size() - 1; ++i){
            
            temp = std::max(temp, std::abs(energy_list[i] - energy_list[i+1]));
        }

        ++t;

    }

    // compute simulation time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end-start);
    int total_secs = duration.count();
    int mins = total_secs / 60;
    int secs = total_secs % 60;
    std::cout << "Execution time: " << mins << ":" << (secs < 10 ? "0" : "") << secs << std::endl;

    return model.state(); 
}