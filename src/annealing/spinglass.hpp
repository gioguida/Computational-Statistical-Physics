// File: src/annealing/spinglass.hpp
// Purpose: Simulated annealing solver and related native interfaces.
// Usage: Build and invoke from C++ binaries or Python orchestration layers.

#ifndef SPINGLASS_HPP
#define SPINGLASS_HPP

#include "../interaction/interaction.hpp"

#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

        // Curvature triplet parameters
        double curvature_bonus_;
        double curvature_penalty_;
        double curvature_tolerance_;
        bool use_curvature_;

        // Segment geometry indexed by segment id (= spin index)
        std::vector<double> dx_;
        std::vector<double> dy_;
        std::vector<int> hit_a_;
        std::vector<int> hit_b_;

        // hit_id -> segment indices with hit_a == hit_id (segment starts at that hit)
        std::unordered_map<int, std::vector<int>> feeds_into_;
        // hit_id -> segment indices with hit_b == hit_id (segment ends at that hit)
        std::unordered_map<int, std::vector<int>> fed_by_;

        double triplet_energy(int j, int i, int k) const;
        double compute_triplet_delta(int site) const;

    public:
        Spinglass(int N, interaction_mat_t J, std::vector<double> h, int seed,
                  const std::vector<Segment>& segments = {},
                  double curvature_bonus = 0.0,
                  double curvature_penalty = 0.0,
                  double curvature_tolerance = 0.08);

        void set_T(double const& T);

        void compute_initial_energy();
        double get_energy() const;
        std::vector<int> state() const;

        void flip_site(int const& site);
        double deltaE(int const& site) const;
        MetropolisStepStats step();

};
#endif
