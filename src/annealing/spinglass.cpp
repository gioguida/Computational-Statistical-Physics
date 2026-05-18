// File: src/annealing/spinglass.cpp
// Purpose: Simulated annealing solver and related native interfaces.
// Usage: Build and invoke from C++ binaries or Python orchestration layers.

#include "spinglass.hpp"
#include <cmath>


Spinglass::Spinglass(int N, interaction_mat_t J, std::vector<double> h, int seed,
                     const std::vector<Segment>& segments,
                     double curvature_bonus, double curvature_penalty, double curvature_tolerance)
    : N_(N), J_(J), h_(N_), configuration_(N_), rng_(seed),
      curvature_bonus_(curvature_bonus),
      curvature_penalty_(curvature_penalty),
      curvature_tolerance_(curvature_tolerance),
      use_curvature_(curvature_bonus != 0.0 || curvature_penalty != 0.0),
      dx_(N, 0.0), dy_(N, 0.0), hit_a_(N, -1), hit_b_(N, -1)
{
    if (h.size() != static_cast<std::size_t>(N_)) {
        throw std::invalid_argument("External field size does not match number of spins");
    }

    std::bernoulli_distribution bernoulli(0.5);
    for (int i = 0; i < N_; ++i)
        configuration_[i] = bernoulli(rng_);

    h_ = h;

    // Build geometry arrays and adjacency maps from segment data.
    for (int seg_id = 0; seg_id < static_cast<int>(segments.size()) && seg_id < N_; ++seg_id) {
        const Segment& s = segments[seg_id];
        dx_[seg_id]    = s.dx;
        dy_[seg_id]    = s.dy;
        hit_a_[seg_id] = s.hit_a;
        hit_b_[seg_id] = s.hit_b;
        feeds_into_[s.hit_a].push_back(seg_id);
        fed_by_[s.hit_b].push_back(seg_id);
    }
}


void Spinglass::set_T(double const& T) {
    T_ = T;
}


// Energy contribution of the ordered triplet (j → i → k).
double Spinglass::triplet_energy(int j, int i, int k) const {
    const double cross_ji = dx_[j] * dy_[i] - dy_[j] * dx_[i];
    const double dot_ji   = dx_[j] * dx_[i] + dy_[j] * dy_[i];
    const double kappa_1  = std::atan2(cross_ji, dot_ji);

    const double cross_ik = dx_[i] * dy_[k] - dy_[i] * dx_[k];
    const double dot_ik   = dx_[i] * dx_[k] + dy_[i] * dy_[k];
    const double kappa_2  = std::atan2(cross_ik, dot_ik);

    if (std::fabs(kappa_1 - kappa_2) < curvature_tolerance_)
        return -curvature_bonus_;
    return curvature_penalty_;
}


// Net triplet energy change when spin `site` is flipped.
// Returns (1 - 2*x_i) * sum_of_triplet_energies_involving_site.
double Spinglass::compute_triplet_delta(int site) const {
    if (!use_curvature_) return 0.0;

    const int ha_i = hit_a_[site];
    const int hb_i = hit_b_[site];

    const auto it_fb = fed_by_.find(ha_i);     // segments ending where site starts
    const auto it_fi = feeds_into_.find(hb_i); // segments starting where site ends

    double sum = 0.0;

    // Role: middle — (j → site → k)
    if (it_fb != fed_by_.end() && it_fi != feeds_into_.end()) {
        for (int j : it_fb->second) {
            if (j == site || configuration_[j] == 0) continue;
            for (int k : it_fi->second) {
                if (k == site || k == j || configuration_[k] == 0) continue;
                sum += triplet_energy(j, site, k);
            }
        }
    }

    // Role: first — (site → m → k)
    if (it_fi != feeds_into_.end()) {
        for (int m : it_fi->second) {
            if (m == site || configuration_[m] == 0) continue;
            const auto it_m = feeds_into_.find(hit_b_[m]);
            if (it_m == feeds_into_.end()) continue;
            for (int k : it_m->second) {
                if (k == site || k == m || configuration_[k] == 0) continue;
                sum += triplet_energy(site, m, k);
            }
        }
    }

    // Role: last — (j → m → site)
    if (it_fb != fed_by_.end()) {
        for (int m : it_fb->second) {
            if (m == site || configuration_[m] == 0) continue;
            const auto it_m = fed_by_.find(hit_a_[m]);
            if (it_m == fed_by_.end()) continue;
            for (int j : it_m->second) {
                if (j == site || j == m || configuration_[j] == 0) continue;
                sum += triplet_energy(j, m, site);
            }
        }
    }

    return (1 - 2 * configuration_[site]) * sum;
}


void Spinglass::compute_initial_energy() {
    double energy = 0.0;

    for (int i = 0; i < N_; ++i) {
        for (int k = 0; k < static_cast<int>(J_[i].size()); ++k) {
            int j = J_[i][k].first;
            energy -= 0.5 * J_[i][k].second * configuration_[i] * configuration_[j];
        }
        energy -= h_[i] * configuration_[i];
    }

    // Triplet contribution — enumerate each triplet exactly once via the middle spin.
    if (use_curvature_) {
        for (int i = 0; i < N_; ++i) {
            if (configuration_[i] == 0) continue;
            const auto it_fb = fed_by_.find(hit_a_[i]);
            const auto it_fi = feeds_into_.find(hit_b_[i]);
            if (it_fb == fed_by_.end() || it_fi == feeds_into_.end()) continue;
            for (int j : it_fb->second) {
                if (j == i || configuration_[j] == 0) continue;
                for (int k : it_fi->second) {
                    if (k == i || k == j || configuration_[k] == 0) continue;
                    energy += triplet_energy(j, i, k);
                }
            }
        }
    }

    E_ = energy;
}


double Spinglass::get_energy() const {
    return E_;
}


std::vector<int> Spinglass::state() const {
    return configuration_;
}


void Spinglass::flip_site(int const& site) {
    configuration_[site] = 1 - configuration_[site];
}


double Spinglass::deltaE(int const& site) const {
    int x_i = configuration_[site];
    double f_i = 0.;

    for (int k = 0; k < static_cast<int>(J_[site].size()); ++k) {
        f_i += J_[site][k].second * configuration_[J_[site][k].first];
    }
    f_i += h_[site];

    const double triplet_delta = compute_triplet_delta(site);
    return -(1 - 2 * x_i) * f_i + triplet_delta;
}


MetropolisStepStats Spinglass::step() {
    MetropolisStepStats stats;
    stats.proposed = 1;

    std::uniform_int_distribution<> unif_int(0, N_ - 1);
    int site = unif_int(rng_);

    double dE = deltaE(site);
    stats.deltaE = dE;
    if (dE <= 0) {
        flip_site(site);
        E_ = E_ + dE;
        stats.accepted = 1;
    } else {
        std::uniform_real_distribution<> unif_real(0.0, 1.0);
        double prob = std::exp(-dE / T_);
        if (unif_real(rng_) < prob) {
            flip_site(site);
            E_ = E_ + dE;
            stats.accepted = 1;
        }
    }
    return stats;
}
