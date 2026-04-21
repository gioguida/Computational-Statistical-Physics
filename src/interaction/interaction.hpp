#ifndef INTERACTION_HPP
#define INTERACTION_HPP

#include <vector>
#include <map>
#include <stdlib.h>
#include <cmath>

using hit_vec_t = std::vector<Hit>;
using hit_group_t = std::map<int,std::vector<Hit>>;
using seg_vec_t = std::vector<Segment>;
using interaction_mat_t = std::vector<std::vector<std::pair<int,double>>>;

struct Hit {
    int id;
    int layer_id;
    double layer_radius;
    double x , y;

    Hit(int, int, double, double, double);
};

struct Segment {
    int id;
    int hit_a;
    int hit_b;
    int layer_a;
    int layer_b;
    double dx, dy;
    double angle;

    Segment(Hit, int, Hit, int, int);
};

double dot(Segment, Segment);

double cos(Segment, Segment);

hit_vec_t read_hits();

hit_group_t group_hits_by_layer(hit_vec_t);

seg_vec_t create_segments(hit_group_t);

interaction_mat_t interaction_matrix(seg_vec_t);



#endif