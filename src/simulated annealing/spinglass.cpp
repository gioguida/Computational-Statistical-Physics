#include"spinglass.hpp"


void Spinglass::set_T(double const& T) {
    T_ = T;
};


void Spinglass::compute_initial_energy() {
    int hamiltonian = 0;
    int total_sum = 0;

    // Neighbouring pairs products with periodic BC
    for(int i = 0; i < N_; ++i) {
        for(int k = 0; k < J_[i].size(); ++k) {
            int j = J_[i][k].first;
            // H = -sum_ij J_ij * s_i * s_j - sum_i h_i * s_i
            hamiltonian += J_[i][k].second * configuration_[i] * configuration_[j];
        }
        hamiltonian -= h_[i] * configuration_[i];
    }
};


double Spinglass::get_energy() const {
    return E_;
}


std::vector<int> Spinglass::state() const {
    return configuration_;
}


void Spinglass::flip_site(int const& site) {
        configuration_[site] *= -1;
};


double Spinglass::deltaE(int const& site) const {
    int s_i = configuration_[site];
    double dE = 0.;

    for(int k = 0; k < J_[site].size(); ++k) {
        dE += (-2 * s_i) * configuration_[J_[site][k].first];
    }

    dE += 2*h_[site]* s_i; 
    return dE;
};


void Spinglass::step() {
    // randomly select new site
    std::uniform_int_distribution<> unif_int(0, N_*N_-1);
    int site = unif_int(rng_);

    // compute the energy difference
    double dE = deltaE(site);
    // accept the move if energy decreases
    if(dE <= 0) {
        flip_site(site);
        // update energy
        E_ = E_ + dE;
    } else { // play the Metropolis game
        // extract a random number
        std::uniform_real_distribution<> unif_real(0.0, 1.0);
        double prob = std::exp(-dE/(T_));
        if(unif_real(rng_) < prob) {
            flip_site(site);
            // update energy
            E_ = E_ + dE;
        }
    } 
};
