#include"annealing.hpp"
#include<list>
#include <algorithm>



#define N_energy 5

namespace {

int count_selected_spins(const std::vector<int>& state) {
    return static_cast<int>(std::count_if(state.begin(), state.end(), [](int spin) {
        return spin > 0;
    }));
}

}

AnnealingResult main_simulation(int N, interaction_mat_t J, std::vector<double> h,
    double T_min, double T_max, double T_step, double toll,
    int N_sweeps, int seed, int log_every_steps) {
    // set simulation parameters
    int N_temp = static_cast<int>((T_max - T_min)/T_step);

    std::filesystem::create_directory("../../results");
    
    std::cout << "--- Starting simulation ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Express smulation iterations in sweeps
    int sweep = N;
    int N_eq = N_sweeps * sweep;
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

    AnnealingResult result;
    result.trace.push_back({
        -1,
        T_max,
        model.get_energy(),
        count_selected_spins(model.state())
    });

    std::cout << "--- Beginning Simulation ---" << std::endl;

    std::list<double> energy_list;
    int t = 0;
    double dev = toll + 1;

    while(t < N_temp && toll < dev ) {

        double T = (std::cos(t * M_PI / N_temp) + 1) * 0.5 * (T_max - T_min) + T_min;
    
        model.set_T(T);

        if (t % log_every_steps == 0) {
            std::cout << "\n--- Step " << t << " of " << N_temp <<" ---\n";
            std::cout << " Temperature = " << T << std::endl;
        }


        // equilibrate the model
        for(int i = 0; i < N_eq; ++i) {
            model.step();
        }

        const std::vector<int> current_state = model.state();
        result.trace.push_back({
            t,
            T,
            model.get_energy(),
            count_selected_spins(current_state)
        });
        
        energy_list.push_back(model.get_energy());
        if (t > N_energy)   energy_list.pop_front();

        double temp = toll + 1;
        if (energy_list.size() >= 2) {
            temp = 0;
            for (auto it = energy_list.begin(); std::next(it) != energy_list.end(); ++it){
                temp = std::max(temp, std::abs(*it - *(std::next(it))));
            }
        }
        dev = temp;

        ++t;

    }

    // compute simulation time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end-start);
    int total_secs = duration.count();
    int mins = total_secs / 60;
    int secs = total_secs % 60;
    std::cout << "Execution time: " << mins << ":" << (secs < 10 ? "0" : "") << secs << std::endl;

    result.state = model.state();
    return result;
}
