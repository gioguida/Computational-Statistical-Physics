#ifndef SPINGLASS_HPP
#define SPINGLASS_HPP

#include "../interaction/interaction.hpp"

#include <random>
#include <stdexcept>

struct MetropolisStepStats {
    int proposed = 0;
    int accepted = 0;
    double deltaE = 0.0;
};

class Spinglass {
    private:
        int const N_;
        interaction_mat_t J_;
        std::vector<double> h_;
        double T_;
        double E_;
        std::vector<int> configuration_;
        std::mt19937 rng_;

    public:
        Spinglass(int N, interaction_mat_t J, std::vector<double> h, int seed):
         N_(N), J_(J), h_(N_), configuration_(N_), rng_(seed) {
            if (h.size() != static_cast<std::size_t>(N_)) {
                throw std::invalid_argument("External field size does not match number of spins");
            }
            
            // Initialize from the configured RNG so repeated runs with the same seed are reproducible.
            std::bernoulli_distribution bernoulli(0.5);

            for(int i = 0; i < N_; ++i)
                configuration_[i] = 2*bernoulli(rng_) - 1;

            h_ = h;
        };

        void set_T(double const& T);

        void compute_initial_energy();
        double get_energy() const;
        std::vector<int> state() const;

        void flip_site(int const& site);
        double deltaE(int const& site) const;
        MetropolisStepStats step();

};
#endif
