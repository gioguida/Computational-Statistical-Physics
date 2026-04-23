#include "../src/interaction/interaction.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
	std::string hits_csv;
	std::string out_dir;
	double theta_max = std::acos(-1.0) / 6.0;  // 30 deg
	double penalty = 10.0;
};

std::string require_value(int argc, char** argv, int& i) {
	if (i + 1 >= argc) {
		throw std::runtime_error(std::string("Missing value for argument: ") + argv[i]);
	}
	++i;
	return argv[i];
}

Config parse_args(int argc, char** argv) {
	Config cfg;

	// Keep CLI intentionally small so a Python control panel can call this reliably.
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "--hits-csv") {
			cfg.hits_csv = require_value(argc, argv, i);
		} else if (arg == "--out-dir") {
			cfg.out_dir = require_value(argc, argv, i);
		} else if (arg == "--theta-max") {
			cfg.theta_max = std::stod(require_value(argc, argv, i));
		} else if (arg == "--penalty") {
			cfg.penalty = std::stod(require_value(argc, argv, i));
		} else if (arg == "--help" || arg == "-h") {
			std::cout
				<< "Usage: run_interaction --hits-csv <path> --out-dir <path> [options]\n"
				<< "Options:\n"
				<< "  --theta-max <float>  Max angular threshold in radians (default: pi/6)\n"
				<< "  --penalty <float>    Penalty for fork/merge conflicts (default: 10)\n";
			std::exit(0);
		} else {
			throw std::runtime_error("Unknown argument: " + arg);
		}
	}

	if (cfg.hits_csv.empty()) {
		throw std::runtime_error("Missing required argument --hits-csv");
	}
	if (cfg.out_dir.empty()) {
		throw std::runtime_error("Missing required argument --out-dir");
	}
	return cfg;
}

hit_vec_t read_hits_from_csv(const std::filesystem::path& path) {
	if (!std::filesystem::exists(path)) {
		throw std::runtime_error("Hits CSV does not exist: " + path.string());
	}

	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open hits CSV: " + path.string());
	}

	hit_vec_t hits;
	std::string line;
	std::getline(file, line);  // skip header

	// Expected order: hit_id, layer_id, layer_radius, hit_x, hit_y[, ...]
	while (std::getline(file, line)) {
		if (line.empty()) {
			continue;
		}

		std::stringstream line_stream(line);
		std::string token;
		std::vector<std::string> tokens;
		while (std::getline(line_stream, token, ',')) {
			tokens.push_back(token);
		}

		if (tokens.size() < 5) {
			throw std::runtime_error("Invalid line in hits CSV: " + line);
		}

		const int hit_id = std::stoi(tokens[0]);
		const int layer_id = std::stoi(tokens[1]);
		const double layer_radius = std::stod(tokens[2]);
		const double hit_x = std::stod(tokens[3]);
		const double hit_y = std::stod(tokens[4]);
		hits.emplace_back(hit_id, layer_id, layer_radius, hit_x, hit_y);
	}

	return hits;
}

void write_segments_csv(const std::filesystem::path& out_path, const seg_vec_t& segments) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "seg_id,hit_a,hit_b,layer_a,layer_b,dx,dy,angle\n";
	out << std::setprecision(17);
	for (const auto& s : segments) {
		out << s.id << ',' << s.hit_a << ',' << s.hit_b << ',' << s.layer_a << ',' << s.layer_b << ',' << s.dx
			<< ',' << s.dy << ',' << s.angle << '\n';
	}
}

void write_edges_csv(const std::filesystem::path& out_path, const interaction_mat_t& J) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "i,j,Jij\n";
	out << std::setprecision(17);
	// Persist only upper-triangle edges to avoid duplicating undirected pairs.
	for (int i = 0; i < static_cast<int>(J.size()); ++i) {
		for (const auto& [j, val] : J[i]) {
			if (j > i) {
				out << i << ',' << j << ',' << val << '\n';
			}
		}
	}
}

void write_meta_json(const std::filesystem::path& out_path,
					 const Config& cfg,
					 int n_hits,
					 int n_layers,
					 int n_segments,
					 int n_edges) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "{\n"
		<< "  \"hits_csv\": \"" << cfg.hits_csv << "\",\n"
		<< "  \"theta_max\": " << cfg.theta_max << ",\n"
		<< "  \"penalty\": " << cfg.penalty << ",\n"
		<< "  \"n_hits\": " << n_hits << ",\n"
		<< "  \"n_layers\": " << n_layers << ",\n"
		<< "  \"n_segments\": " << n_segments << ",\n"
		<< "  \"n_nonzero_edges\": " << n_edges << "\n"
		<< "}\n";
}

}  // namespace

int main(int argc, char** argv) {
	try {
		const Config cfg = parse_args(argc, argv);

		std::cout << "--- Interaction Runner ---\n";
		// Stage pipeline: hits -> grouped hits -> segments -> sparse interaction matrix.
		hit_vec_t hits = read_hits_from_csv(cfg.hits_csv);
		hit_group_t grouped = group_hits_by_layer(hits);
		seg_vec_t segments = create_segments(grouped);
		interaction_mat_t J = interaction_matrix(segments, cfg.theta_max, cfg.penalty);

		int n_edges = 0;
		for (int i = 0; i < static_cast<int>(J.size()); ++i) {
			for (const auto& [j, _] : J[i]) {
				if (j > i) {
					++n_edges;
				}
			}
		}

		const std::filesystem::path out_dir(cfg.out_dir);
		std::filesystem::create_directories(out_dir);

		write_segments_csv(out_dir / "segments.csv", segments);
		write_edges_csv(out_dir / "J_edges.csv", J);
		write_meta_json(out_dir / "interaction_meta.json",
						cfg,
						static_cast<int>(hits.size()),
						static_cast<int>(grouped.size()),
						static_cast<int>(segments.size()),
						n_edges);

		std::cout << "Saved interaction artifacts to: " << out_dir << "\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}
}

