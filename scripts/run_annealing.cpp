#include "../src/annealing/annealing.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
	std::string segments_csv;
	std::string edges_csv;
	std::string out_dir;
	double t_min = 0.05;
	double t_max = 5.0;
	double t_step = 0.05;
	double toll = 1e-3;
	int eq_sweeps = 50;
	int seed = 123;
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
	// Keep argument names explicit so orchestration scripts can pass them unambiguously.
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg == "--segments-csv") {
			cfg.segments_csv = require_value(argc, argv, i);
		} else if (arg == "--edges-csv") {
			cfg.edges_csv = require_value(argc, argv, i);
		} else if (arg == "--out-dir") {
			cfg.out_dir = require_value(argc, argv, i);
		} else if (arg == "--t-min") {
			cfg.t_min = std::stod(require_value(argc, argv, i));
		} else if (arg == "--t-max") {
			cfg.t_max = std::stod(require_value(argc, argv, i));
		} else if (arg == "--t-step") {
			cfg.t_step = std::stod(require_value(argc, argv, i));
		} else if (arg == "--toll") {
			cfg.toll = std::stod(require_value(argc, argv, i));
		} else if (arg == "--eq-sweeps") {
			cfg.eq_sweeps = std::stoi(require_value(argc, argv, i));
		} else if (arg == "--seed") {
			cfg.seed = std::stoi(require_value(argc, argv, i));
		} else if (arg == "--help" || arg == "-h") {
			std::cout
				<< "Usage: run_annealing --segments-csv <path> --edges-csv <path> --out-dir <path> [options]\n"
				<< "Options:\n"
				<< "  --t-min <float>      Minimum temperature (default: 0.05)\n"
				<< "  --t-max <float>      Maximum temperature (default: 5.0)\n"
				<< "  --t-step <float>     Temperature step (default: 0.05)\n"
				<< "  --toll <float>       Energy convergence tolerance (default: 1e-3)\n"
				<< "  --eq-sweeps <int>    Equilibration sweeps per temperature (default: 50)\n"
				<< "  --seed <int>         RNG seed (default: 123)\n";
			std::exit(0);
		} else {
			throw std::runtime_error("Unknown argument: " + arg);
		}
	}

	if (cfg.segments_csv.empty()) {
		throw std::runtime_error("Missing required argument --segments-csv");
	}
	if (cfg.edges_csv.empty()) {
		throw std::runtime_error("Missing required argument --edges-csv");
	}
	if (cfg.out_dir.empty()) {
		throw std::runtime_error("Missing required argument --out-dir");
	}
	if (cfg.t_max <= cfg.t_min) {
		throw std::runtime_error("Invalid temperature range: require t_max > t_min");
	}
	if (cfg.t_step <= 0.0) {
		throw std::runtime_error("Invalid temperature step: require t_step > 0");
	}
	if (cfg.toll <= 0.0) {
		throw std::runtime_error("Invalid tolerance: require toll > 0");
	}
	if (cfg.eq_sweeps <= 0) {
		throw std::runtime_error("Sweep count must be positive");
	}

	return cfg;
}

std::vector<double> read_segment_lengths_csv(const std::filesystem::path& segments_csv) {
	if (!std::filesystem::exists(segments_csv)) {
		throw std::runtime_error("Segments CSV does not exist: " + segments_csv.string());
	}

	std::ifstream file(segments_csv);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open segments CSV: " + segments_csv.string());
	}

	std::string line;
	std::getline(file, line);  // header

	int max_seg_id = -1;
	std::vector<std::pair<int, double>> parsed_segments;
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

		if (tokens.size() < 7) {
			throw std::runtime_error("Invalid line in segments CSV: " + line);
		}

		const int seg_id = std::stoi(tokens[0]);
		const double dx = std::stod(tokens[5]);
		const double dy = std::stod(tokens[6]);
		const double length = std::sqrt(dx * dx + dy * dy);
		parsed_segments.emplace_back(seg_id, length);

		if (seg_id > max_seg_id) {
			max_seg_id = seg_id;
		}
	}

	if (max_seg_id < 0) {
		throw std::runtime_error("No segments found in CSV: " + segments_csv.string());
	}

	std::vector<double> h(max_seg_id + 1, std::numeric_limits<double>::quiet_NaN());
	for (const auto& [seg_id, length] : parsed_segments) {
		if (seg_id < 0 || seg_id >= static_cast<int>(h.size())) {
			throw std::runtime_error("Segment id out of range in segments CSV");
		}
		if (!std::isnan(h[seg_id])) {
			throw std::runtime_error("Duplicate segment id in segments CSV: " + std::to_string(seg_id));
		}
		h[seg_id] = length;
	}

	for (int seg_id = 0; seg_id < static_cast<int>(h.size()); ++seg_id) {
		if (std::isnan(h[seg_id])) {
			throw std::runtime_error("Missing segment id in segments CSV: " + std::to_string(seg_id));
		}
	}

	return h;
}

interaction_mat_t read_edges_csv(const std::filesystem::path& edges_csv, int N) {
	if (!std::filesystem::exists(edges_csv)) {
		throw std::runtime_error("Edges CSV does not exist: " + edges_csv.string());
	}

	std::ifstream file(edges_csv);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open edges CSV: " + edges_csv.string());
	}

	interaction_mat_t J(N);
	std::string line;
	std::getline(file, line);  // header

	// Rebuild a symmetric adjacency list from the stored sparse edge list.
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

		if (tokens.size() < 3) {
			throw std::runtime_error("Invalid line in edges CSV: " + line);
		}

		const int i = std::stoi(tokens[0]);
		const int j = std::stoi(tokens[1]);
		const double Jij = std::stod(tokens[2]);

		if (i < 0 || j < 0 || i >= N || j >= N) {
			throw std::runtime_error("Edge index out of range in edges CSV: " + line);
		}
		if (i == j) {
			continue;
		}

		J[i].push_back(std::make_pair(j, Jij));
		J[j].push_back(std::make_pair(i, Jij));
	}

	return J;
}

void write_final_state_csv(const std::filesystem::path& out_path, const std::vector<int>& state) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "seg_id,spin,selected\n";
	for (int i = 0; i < static_cast<int>(state.size()); ++i) {
		out << i << ',' << state[i] << ',' << (state[i] > 0 ? 1 : 0) << '\n';
	}
}

void write_meta_json(const std::filesystem::path& out_path,
					 const Config& cfg,
					 int N,
					 int n_edges_undirected) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "{\n"
		<< "  \"segments_csv\": \"" << cfg.segments_csv << "\",\n"
		<< "  \"edges_csv\": \"" << cfg.edges_csv << "\",\n"
		<< "  \"N\": " << N << ",\n"
		<< "  \"n_edges_undirected\": " << n_edges_undirected << ",\n"
		<< "  \"t_min\": " << cfg.t_min << ",\n"
		<< "  \"t_max\": " << cfg.t_max << ",\n"
		<< "  \"t_step\": " << cfg.t_step << ",\n"
		<< "  \"toll\": " << cfg.toll << ",\n"
		<< "  \"eq_sweeps\": " << cfg.eq_sweeps << ",\n"
		<< "  \"seed\": " << cfg.seed << "\n"
		<< "}\n";
}

int count_undirected_edges(const interaction_mat_t& J) {
	int n_edges = 0;
	for (int i = 0; i < static_cast<int>(J.size()); ++i) {
		for (const auto& [j, _] : J[i]) {
			if (j > i) {
				++n_edges;
			}
		}
	}
	return n_edges;
}

}  // namespace

int main(int argc, char** argv) {
	try {
		const Config cfg = parse_args(argc, argv);

		std::cout << "--- Annealing Runner ---\n";
		// Infer spin count and local fields h_i from segment lengths.
		std::vector<double> h = read_segment_lengths_csv(cfg.segments_csv);
		const int N = static_cast<int>(h.size());
		interaction_mat_t J = read_edges_csv(cfg.edges_csv, N);

		// Run annealing using h_i = segment length.
		std::vector<int> state = main_simulation(N,
												 J,
												 h,
												 cfg.t_min,
												 cfg.t_max,
												 cfg.t_step,
												 cfg.toll,
												 cfg.eq_sweeps,
												 cfg.seed);

		const std::filesystem::path out_dir(cfg.out_dir);
		std::filesystem::create_directories(out_dir);

		// Persist final spin assignment and run metadata for downstream analysis.
		write_final_state_csv(out_dir / "final_state.csv", state);
		write_meta_json(out_dir / "annealing_meta.json", cfg, N, count_undirected_edges(J));

		std::cout << "Saved annealing artifacts to: " << out_dir << "\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}
}
