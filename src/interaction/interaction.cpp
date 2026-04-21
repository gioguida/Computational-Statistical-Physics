#include "interaction.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <stdexcept>


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
    /* TODO */
}


double cos(Segment s_i, Segment s_j) {
    /* TODO */
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
    int N_segments = 0;
    while(grouped.count(n_layers) > 0) {
        n_layers++;
        N_segments += grouped[n_layers-1].size() * grouped[n_layers].size();
    }

    segments.reserve(N_segments);
    int seg_count = 0;

    for(int l = 0; l < n_layers - 1; ++l) {
        int count_a = 0;
        int count_b = 0;
        for(auto const& hit_a : grouped[l]) {
            for(auto const& hit_b : grouped[l+1]) {
                segments.emplace_back(hit_a, count_a, hit_b, count_b, seg_count);
                count_b++;
                seg_count++;
            }
            count_a++;
        }
    }   
}

interaction_mat_t interaction_matrix(seg_vec_t segments) {
    interaction_mat_t J;
    int N_segments = segments.size();

    for(int i=0; i < N_segments - 1; ++i) {
        std::vector<std::pair<int, double>> J_i;
        for(int j=i+1; j < N_segments; ++j) {
            if((segments[i].layer_b == segments[j].layer_a) &&
            (segments[i].hit_b == segments[j].hit_a)) {
                // implement reward computation
                double reward = 1.0;
                J_i.emplace_back(std::pair<int,double>{j, reward});
            }
        }
        J.push_back(J_i);
    }
}





int main() {
    hit_vec_t hits = read_hits();
    hit_group_t grouped = group_hits_by_layer(hits);

    std::cout << grouped[0].size() << std::endl;
    std::cout << grouped[1].size() << std::endl;
    std::cout << grouped[2].size() << std::endl;
    std::cout << grouped[3].size() << std::endl;
    std::cout << grouped[4].size() << std::endl;
}
