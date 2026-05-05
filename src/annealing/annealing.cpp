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
    double T_min, double T_max, int N_steps, double toll,
    int N_sweeps, int seed, int log_every_steps, int checkpoint_every_steps) {
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
    result.best_state = model.state();
    result.best_energy = model.get_energy();
    result.trace.push_back({
        -1,
        T_max,
        model.get_energy(),
        count_selected_spins(model.state())
    });
    result.checkpoints.push_back({
        -1,
        T_max,
        model.get_energy(),
        model.state()
    });

    std::cout << "--- Beginning Simulation ---" << std::endl;

    std::list<double> energy_list;
    int t = 0;
    double dev = toll + 1;

    while(t < N_steps && toll < dev ) {

        double exponent = std::log(N_steps - t) / std::log(N_steps);
        double T = T_min * std::pow(T_max / T_min, exponent);
    
        model.set_T(T);

        if (t % log_every_steps == 0) {
            std::cout << "\n--- Step " << t << " of " << N_steps <<" ---\n";
            std::cout << " Temperature = " << T << std::endl;
        }


        // equilibrate the model
        for(int i = 0; i < N_eq; ++i) {
            model.step();
        }

        const std::vector<int> current_state = model.state();
        const double current_energy = model.get_energy();
        if (current_energy < result.best_energy) {
            result.best_energy = current_energy;
            result.best_state = current_state;
        }
        result.trace.push_back({
            t,
            T,
            current_energy,
            count_selected_spins(current_state)
        });
        if (t % checkpoint_every_steps == 0) {
            result.checkpoints.push_back({
                t,
                T,
                current_energy,
                current_state
            });
        }
        
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
    const AnnealingTraceSample& last_sample = result.trace.back();
    if (result.checkpoints.empty() || result.checkpoints.back().step != last_sample.step) {
        result.checkpoints.push_back({
            last_sample.step,
            last_sample.temperature,
            last_sample.energy,
            result.state
        });
    }
    return result;
}
