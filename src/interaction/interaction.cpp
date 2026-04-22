#include "interaction.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <stdexcept>
#include <iomanip>


Hit::Hit(int idx, int layer_idx, double layer_r, double x_hit, double y_hit)
: id(idx), layer_id(layer_idx), layer_radius(layer_r), x(x_hit), y(y_hit) {}


Segment::Segment(Hit a, int a_id, Hit b, int b_id, int seg_idx)
: hit_a(a_id), layer_a(a.layer_id), hit_b(b_id), layer_b(b.layer_id), id(seg_idx) {
    if(a.layer_id > b.layer_id) {
        dx = a.x - b.x;
        dy = a.y - b.y;
    } else {
        dx = b.x - a.x;
        dy = b.y - a.y;
    }

    angle = std::atan2(dy,dx);
}


double dot(Segment s_i, Segment s_j) {
    double prod_x = s_i.dx * s_j.dx; 
    double prod_y = s_i.dy * s_j.dy;
    
    return prod_x + prod_y;
}


double seg_alignment(Segment s_i, Segment s_j) {
    double dot_prod = dot(s_i, s_j);
    double norm_i = std::sqrt(dot(s_i,s_i));
    double norm_j = std::sqrt(dot(s_j,s_j));
    return dot_prod / (norm_i * norm_j);
}


hit_vec_t read_hits() {
    const std::filesystem::path data_path{"../../data/training_hits.csv"};

    if(!std::filesystem::exists(data_path)) {
        throw std::runtime_error("Data file does not exist");
    }

    std::ifstream file(data_path);
    if(!file.is_open()) {
        throw std::runtime_error("Could not open data file");
    }

    hit_vec_t hits;
    std::string line;

    std::getline(file, line);

    while(std::getline(file, line)) {
        if(line.empty()) {
            continue;
        }

        std::stringstream line_stream(line);
        std::string token;

        std::getline(line_stream, token, ',');
        const int hit_id = std::stoi(token);

        std::getline(line_stream, token, ',');
        const int layer_id = std::stoi(token);

        std::getline(line_stream, token, ',');
        const double layer_radius = std::stod(token);

        std::getline(line_stream, token, ',');
        const double hit_x = std::stod(token);

        std::getline(line_stream, token, ',');
        const double hit_y = std::stod(token);

        hits.emplace_back(hit_id, layer_id, layer_radius, hit_x, hit_y);
    }

    return hits;
}


hit_group_t group_hits_by_layer(hit_vec_t hits) {
    hit_group_t grouped_hits;

    for(auto const& h: hits) {
        int key = h.layer_id;
        // magari fare un conto preliminare per efficienza
        if(grouped_hits.count(key)==0) {
            grouped_hits[key] = hit_vec_t{};
        }
        grouped_hits[key].push_back(h);
    }

    return grouped_hits;
};


seg_vec_t create_segments(hit_group_t grouped) {
    seg_vec_t segments;

    // find number of detector layers
    int n_layers = 0;
    while(grouped.count(n_layers) > 0) {
        ++n_layers;
    }
    
    // Calculate total number of segments
    int N_segments = 0;
    for(int l = 0; l < n_layers - 1; ++l) {
        N_segments += grouped[l].size() * grouped[l+1].size();
    }

    segments.reserve(N_segments);
    int seg_count = 0;

    for(int l = 0; l < n_layers - 1; ++l) {
        for(auto const& hit_a : grouped[l]) {
            for(auto const& hit_b : grouped[l+1]) {
                segments.emplace_back(hit_a, hit_a.id, hit_b, hit_b.id, seg_count); 
                seg_count++;
            }
        }
    }  
    
    return segments;
}

interaction_mat_t interaction_matrix(seg_vec_t segments, double theta_max, double penalty) {
    int N_segments = segments.size();
    interaction_mat_t J(N_segments);

    for(int i = 0; i < N_segments; ++i) {
        std::vector<std::pair<int, double>> J_i;
        for(int j = i + 1; j < N_segments; ++j) {

            
            if( // potentially aligned segments
                ((segments[i].layer_b == segments[j].layer_a) &&
                    (segments[i].hit_b == segments[j].hit_a)) ||
                ((segments[j].layer_b == segments[i].layer_a) &&
                    (segments[j].hit_b == segments[i].hit_a))
            ) {
                double reward = seg_alignment(segments[i], segments[j]);
                // check if the angle is below the threshold
                reward = reward > std::cos(theta_max) ? reward : 0.0;
                J[i].emplace_back(std::pair<int,double>{j, reward});
                J[j].emplace_back(std::pair<int,double>{i, reward});


            } else if( // merged segments
                (segments[i].layer_b == segments[j].layer_b) &&
                    (segments[i].hit_b == segments[j].hit_b)
            ) {
                J[i].emplace_back(std::pair<int,double>{j, -penalty});
                J[j].emplace_back(std::pair<int,double>{i, -penalty});

            } else if( // forked segments
                (segments[i].layer_a == segments[j].layer_a) &&
                    (segments[i].hit_a == segments[j].hit_a)
            ) {
                J[i].emplace_back(std::pair<int,double>{j, -penalty});
                J[j].emplace_back(std::pair<int,double>{i, -penalty});
            }
        }
    }
    return J;
}


int main() {
    double theta_max = std::acos(-1) / 6.0; // 30°
    double lambda = 10;
    hit_vec_t hits = read_hits();
    std::cout << " 1. hits read" << std::endl;
    hit_group_t grouped = group_hits_by_layer(hits);
    std::cout << " 2. groups formed" << std::endl;
    seg_vec_t segments = create_segments(grouped);
    std::cout << " 3. segments created" << std::endl;
    interaction_mat_t J = interaction_matrix(segments, theta_max, lambda);
    std::cout << " 4. Matrix formed" << std::endl;

    // std::cout << grouped[0].size() << std::endl;
    // std::cout << grouped[1].size() << std::endl;
    // std::cout << grouped[2].size() << std::endl;
    // std::cout << grouped[3].size() << std::endl;
    // std::cout << grouped[4].size() << std::endl;

    // Print J matrix (sparse, upper triangle only)
    std::cout << "\n=== J Matrix (nonzero entries) ===\n";
    std::cout << std::setw(6) << "i"
            << std::setw(6) << "j"
            << std::setw(12) << "J_ij"
            << std::setw(14) << "type" << "\n";
    std::cout << std::string(38, '-') << "\n";

    for (int i = 0; i < segments.size(); i++) {
        for (auto& [j, val] : J[i]) {
            if (j <= i) continue;  // upper triangle only

            std::string type;
            if (segments[i].hit_a == segments[j].hit_a ||
                segments[i].hit_b == segments[j].hit_b)
                type = "COMPETING";
            else
                type = "ALIGNED";

            std::cout << std::setw(6) << i
                    << std::setw(6) << j
                    << std::setw(12) << std::fixed << std::setprecision(4) << val
                    << std::setw(14) << type << "\n";
        }
    }
    std::cout << std::string(38, '-') << "\n";
    std::cout << "Total segments: " << segments.size() << "\n";
}
