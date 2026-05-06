#include"annealing.hpp"
#include<list>
#include <algorithm>
#include <cmath>



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


        double sum_energy = 0.0;
        double sum_energy_sq = 0.0;
        double sum_active = 0.0;
        double sum_active_sq = 0.0;
        int n_proposed = 0;
        int n_accepted = 0;
        double sum_deltaE_neg = 0.0;
        int n_deltaE_neg = 0;
        double sum_deltaE_pos = 0.0;
        int n_deltaE_pos = 0;

        // equilibrate at this temperature and collect sweep-level diagnostics
        for(int sweep_idx = 0; sweep_idx < N_sweeps; ++sweep_idx) {
            for(int i = 0; i < sweep; ++i) {
                const MetropolisStepStats stats = model.step();
                n_proposed += stats.proposed;
                n_accepted += stats.accepted;
                if (stats.deltaE < 0.0) {
                    sum_deltaE_neg += stats.deltaE;
                    ++n_deltaE_neg;
                } else if (stats.deltaE > 0.0) {
                    sum_deltaE_pos += stats.deltaE;
                    ++n_deltaE_pos;
                }
            }

            const double energy = model.get_energy();
            const int n_active = count_selected_spins(model.state());
            sum_energy += energy;
            sum_energy_sq += energy * energy;
            sum_active += static_cast<double>(n_active);
            sum_active_sq += static_cast<double>(n_active * n_active);

            if (energy < result.best_energy) {
                result.best_energy = energy;
                result.best_state = model.state();
            }
        }

        const std::vector<int> current_state = model.state();
        const double current_energy = model.get_energy();
        const double sweeps = static_cast<double>(N_sweeps);
        const double H_mean = sum_energy / sweeps;
        const double H_var = std::max(0.0, (sum_energy_sq / sweeps) - (H_mean * H_mean));
        const double n_active_mean = sum_active / sweeps;
        const double n_active_var = std::max(0.0, (sum_active_sq / sweeps) - (n_active_mean * n_active_mean));
        const double acceptance_rate = n_proposed > 0 ? static_cast<double>(n_accepted) / static_cast<double>(n_proposed) : 0.0;
        const double delta_E_mean_neg = n_deltaE_neg > 0 ? sum_deltaE_neg / static_cast<double>(n_deltaE_neg) : 0.0;
        const double delta_E_mean_pos = n_deltaE_pos > 0 ? sum_deltaE_pos / static_cast<double>(n_deltaE_pos) : 0.0;

        result.annealing_trace.push_back({
            t,
            T,
            H_mean,
            H_var,
            T > 0.0 ? H_var / (T * T) : 0.0,
            acceptance_rate,
            result.best_energy,
            n_active_mean,
            std::sqrt(n_active_var),
            delta_E_mean_neg,
            delta_E_mean_pos
        });

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

    result.state = result.best_state;
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
