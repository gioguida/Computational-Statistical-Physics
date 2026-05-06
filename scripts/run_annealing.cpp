#include "../src/annealing/annealing.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
	std::string segments_csv;
	std::string edges_csv;
	std::string hits_csv;
	std::string out_dir;
	double t_min = 0.05;
	double t_max = 5.0;
	int n_steps = 1000;
	double toll = 1e-3;
	double length_penalty = 0.0;
	double layer_radius_penalty = 0.0;
	double layer01_radial_tolerance = 0.0;
	double first_gap = 0.0;
	int eq_sweeps = 50;
	int log_every_steps = 1;
	int checkpoint_every_steps = 10;
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
		} else if (arg == "--hits-csv") {
			cfg.hits_csv = require_value(argc, argv, i);
		} else if (arg == "--out-dir") {
			cfg.out_dir = require_value(argc, argv, i);
		} else if (arg == "--t-min") {
			cfg.t_min = std::stod(require_value(argc, argv, i));
		} else if (arg == "--t-max") {
			cfg.t_max = std::stod(require_value(argc, argv, i));
		} else if (arg == "--n-steps") {
			cfg.n_steps = std::stod(require_value(argc, argv, i));
		} else if (arg == "--toll") {
			cfg.toll = std::stod(require_value(argc, argv, i));
		} else if (arg == "--length-penalty") {
			cfg.length_penalty = std::stod(require_value(argc, argv, i));
		} else if (arg == "--layer-radius-penalty") {
			cfg.layer_radius_penalty = std::stod(require_value(argc, argv, i));
		} else if (arg == "--layer01-radial-tolerance") {
			cfg.layer01_radial_tolerance = std::stod(require_value(argc, argv, i));
		} else if (arg == "--first-gap") {
			cfg.first_gap = std::stod(require_value(argc, argv, i));
		} else if (arg == "--eq-sweeps") {
			cfg.eq_sweeps = std::stoi(require_value(argc, argv, i));
		} else if (arg == "--log-every-steps") {
			cfg.log_every_steps = std::stoi(require_value(argc, argv, i));
		} else if (arg == "--checkpoint-every-steps") {
			cfg.checkpoint_every_steps = std::stoi(require_value(argc, argv, i));
		} else if (arg == "--seed") {
			cfg.seed = std::stoi(require_value(argc, argv, i));
		} else if (arg == "--help" || arg == "-h") {
				std::cout
					<< "Usage: run_annealing --segments-csv <path> --edges-csv <path> --out-dir <path> [options]\n"
					<< "Options:\n"
					<< "  --hits-csv <path>    Hits CSV used to recover layer-0 radial direction (required when radial penalty is enabled)\n"
					<< "  --t-min <float>      Minimum temperature (default: 0.05)\n"
					<< "  --t-max <float>      Maximum temperature (default: 5.0)\n"
					<< "  --t-step <float>     Temperature step (default: 0.05)\n"
					<< "  --toll <float>       Energy convergence tolerance (default: 1e-3)\n"
					<< "  --length-penalty <float>  Binary penalty per unit selected segment length (default: 0)\n"
					<< "  --layer-radius-penalty <float>  Penalty for layer 0->1 segments that violate radial orientation (default: 0)\n"
					<< "  --layer01-radial-tolerance <float>  Max allowed angular deviation from layer-0 radial direction in radians (default: 0)\n"
					<< "  --first-gap <float>  Radius gap between the first two detector layers (default: 0)\n"
					<< "  --eq-sweeps <int>    Equilibration sweeps per temperature (default: 50)\n"
					<< "  --log-every-steps <int>  Log every N annealing temperature steps (default: 1)\n"
				<< "  --checkpoint-every-steps <int>  Save state every N annealing steps (default: 10)\n"
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
	if (cfg.n_steps <= 0.0) {
		throw std::runtime_error("Invalid temperature step: require n_stes > 0");
	}
	if (cfg.toll <= 0.0) {
		throw std::runtime_error("Invalid tolerance: require toll > 0");
	}
	if (cfg.length_penalty < 0.0) {
		throw std::runtime_error("Invalid length penalty: require length_penalty >= 0");
	}
	if (cfg.layer_radius_penalty < 0.0) {
		throw std::runtime_error("Invalid layer radius penalty: require layer_radius_penalty >= 0");
	}
	if (cfg.layer01_radial_tolerance < 0.0) {
		throw std::runtime_error("Invalid layer-0->1 radial tolerance: require layer01_radial_tolerance >= 0");
	}
	if (cfg.first_gap < 0.0) {
		throw std::runtime_error("Invalid first gap: require first_gap >= 0");
	}
	if (cfg.layer_radius_penalty > 0.0 && cfg.layer01_radial_tolerance > 0.0 && cfg.hits_csv.empty()) {
		throw std::runtime_error("Missing required argument --hits-csv when radial layer penalty is enabled");
	}
	if (cfg.eq_sweeps <= 0) {
		throw std::runtime_error("Sweep count must be positive");
	}
	if (cfg.log_every_steps <= 0) {
		throw std::runtime_error("Log step interval must be positive");
	}
	if (cfg.checkpoint_every_steps <= 0) {
		throw std::runtime_error("Checkpoint step interval must be positive");
	}

	return cfg;
}

// struct SegmentInfo {
// 	double length = 0.0;
// 	int layer_a = -1;
// 	int layer_b = -1;
// };

std::vector<Segment> read_segment_info_csv(const std::filesystem::path& segments_csv) {
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
	std::vector<std::pair<int, Segment>> parsed_segments;
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
			const int hit_a = std::stoi(tokens[1]);
			const int hit_b = std::stoi(tokens[2]);
			const int layer_a = std::stoi(tokens[3]);
			const int layer_b = std::stoi(tokens[4]);
			const double dx = std::stod(tokens[5]);
		const double dy = std::stod(tokens[6]);
		parsed_segments.emplace_back(seg_id, Segment{seg_id, hit_a, hit_b, layer_a, layer_b, dx, dy});

		if (seg_id > max_seg_id) {
			max_seg_id = seg_id;
		}
	}

	if (max_seg_id < 0) {
		throw std::runtime_error("No segments found in CSV: " + segments_csv.string());
	}

	std::vector<Segment> segments;
	segments.reserve(max_seg_id + 1);
	std::vector<bool> seen(max_seg_id + 1, false);
	for (const auto& [seg_id, seg] : parsed_segments) {
		if (seg_id < 0 || seg_id > max_seg_id) {
			throw std::runtime_error("Segment id out of range in segments CSV");
		}
		if (seen[seg_id]) {
			throw std::runtime_error("Duplicate segment id in segments CSV: " + std::to_string(seg_id));
		}
		seen[seg_id] = true;
	}

	for (int seg_id = 0; seg_id <= max_seg_id; ++seg_id) {
		if (!seen[seg_id]) {
			throw std::runtime_error("Missing segment id in segments CSV: " + std::to_string(seg_id));
		}
	}

	std::sort(parsed_segments.begin(), parsed_segments.end(),
			  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
	for (int seg_id = 0; seg_id <= max_seg_id; ++seg_id) {
		segments.emplace_back(parsed_segments[seg_id].second);
	}

	return segments;
}

std::map<int, std::pair<double, double>> read_layer0_hit_positions_csv(const std::filesystem::path& hits_csv) {
	if (!std::filesystem::exists(hits_csv)) {
		throw std::runtime_error("Hits CSV does not exist: " + hits_csv.string());
	}

	std::ifstream file(hits_csv);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open hits CSV: " + hits_csv.string());
	}

	std::string line;
	std::getline(file, line);  // header

	std::map<int, std::pair<double, double>> layer0_hit_positions;
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
		if (layer_id != 0) {
			continue;
		}
		const double x = std::stod(tokens[3]);
		const double y = std::stod(tokens[4]);
		layer0_hit_positions.emplace(hit_id, std::make_pair(x, y));
	}

	return layer0_hit_positions;
}

double angular_distance(double a, double b) {
	const double two_pi = 2.0 * std::acos(-1.0);
	double diff = std::fmod(std::fabs(a - b), two_pi);
	if (diff > std::acos(-1.0)) {
		diff = two_pi - diff;
	}
	return diff;
}

interaction_mat_t read_edges_csv(const std::filesystem::path& edges_csv, int N, std::vector<double>& h) {
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
		const double qij = std::stod(tokens[2]);

		if (i < 0 || j < 0 || i >= N || j >= N) {
			throw std::runtime_error("Edge index out of range in edges CSV: " + line);
		}
		if (i == j) {
			continue;
		}

		// The interaction stage writes binary coefficients q_ij for
		// E_binary = -sum q_ij x_i x_j, where x_i=(s_i+1)/2.
		// Expanding into H = -sum J_ij s_i s_j - sum h_i s_i gives
		// J_ij=q_ij/4 and a q_ij/4 field contribution to both endpoints.
		const double Jij = qij / 4.0;
		J[i].push_back(std::make_pair(j, Jij));
		J[j].push_back(std::make_pair(i, Jij));
		h[i] += Jij;
		h[j] += Jij;
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

void write_lowest_energy_state_csv(const std::filesystem::path& out_path, const std::vector<int>& state) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "seg_id,spin,selected\n";
	for (int i = 0; i < static_cast<int>(state.size()); ++i) {
		out << i << ',' << state[i] << ',' << (state[i] > 0 ? 1 : 0) << '\n';
	}
}

void write_energy_trace_csv(const std::filesystem::path& out_path,
							const std::vector<AnnealingTraceSample>& trace) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "step,temperature,energy,n_selected\n";
	for (const AnnealingTraceSample& sample : trace) {
		out << sample.step << ','
			<< sample.temperature << ','
			<< sample.energy << ','
			<< sample.n_selected << '\n';
	}
}

void write_state_checkpoints_csv(const std::filesystem::path& out_path,
								 const std::vector<AnnealingStateCheckpoint>& checkpoints,
								 int N) {
	std::ofstream out(out_path);
	if (!out.is_open()) {
		throw std::runtime_error("Could not open output file: " + out_path.string());
	}

	out << "checkpoint_idx,step,temperature,energy";
	for (int seg_id = 0; seg_id < N; ++seg_id) {
		out << ",spin_" << seg_id;
	}
	out << '\n';

	for (int idx = 0; idx < static_cast<int>(checkpoints.size()); ++idx) {
		const AnnealingStateCheckpoint& checkpoint = checkpoints[idx];
		if (static_cast<int>(checkpoint.state.size()) != N) {
			throw std::runtime_error("Checkpoint state size mismatch while writing CSV");
		}
		out << idx << ','
			<< checkpoint.step << ','
			<< checkpoint.temperature << ','
			<< checkpoint.energy;
		for (int spin : checkpoint.state) {
			out << ',' << spin;
		}
		out << '\n';
	}
}

void write_meta_json(const std::filesystem::path& out_path,
					 const Config& cfg,
					 int N,
					 int n_edges_undirected,
					 int n_trace_samples,
					 int n_checkpoints,
					 double best_energy) {
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
		<< "  \"n_steps\": " << cfg.n_steps << ",\n"
		<< "  \"toll\": " << cfg.toll << ",\n"
		<< "  \"length_penalty\": " << cfg.length_penalty << ",\n"
		<< "  \"layer_radius_penalty\": " << cfg.layer_radius_penalty << ",\n"
		<< "  \"layer01_radial_tolerance\": " << cfg.layer01_radial_tolerance << ",\n"
		<< "  \"first_gap\": " << cfg.first_gap << ",\n"
		<< "  \"eq_sweeps\": " << cfg.eq_sweeps << ",\n"
		<< "  \"log_every_steps\": " << cfg.log_every_steps << ",\n"
		<< "  \"checkpoint_every_steps\": " << cfg.checkpoint_every_steps << ",\n"
		<< "  \"seed\": " << cfg.seed << ",\n"
		<< "  \"best_energy\": " << best_energy << ",\n"
		<< "  \"n_trace_samples\": " << n_trace_samples << ",\n"
		<< "  \"n_checkpoints\": " << n_checkpoints << "\n"
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
		// Infer spin count and binary-objective local fields from interaction coefficients.
			const std::vector<Segment> segments = read_segment_info_csv(cfg.segments_csv);
			const int N = static_cast<int>(segments.size());
			std::vector<double> h(N, 0.0);
			interaction_mat_t J = read_edges_csv(cfg.edges_csv, N, h);
			std::map<int, std::pair<double, double>> layer0_hit_positions;
			if (cfg.layer_radius_penalty > 0.0 && cfg.layer01_radial_tolerance > 0.0) {
				layer0_hit_positions = read_layer0_hit_positions_csv(cfg.hits_csv);
			}
			for (int i = 0; i < N; ++i) {
				const double seg_len = std::sqrt(segments[i].dx * segments[i].dx + segments[i].dy * segments[i].dy);
				h[i] -= 0.5 * cfg.length_penalty * seg_len;

				const bool is_layer_0_to_1 = (segments[i].layer_a == 0 && segments[i].layer_b == 1);
				if (is_layer_0_to_1 && cfg.layer_radius_penalty > 0.0 && cfg.layer01_radial_tolerance > 0.0) {
					const auto hit_it = layer0_hit_positions.find(segments[i].hit_a);
					if (hit_it == layer0_hit_positions.end()) {
						throw std::runtime_error("Layer-0 hit id missing from hits CSV: " + std::to_string(segments[i].hit_a));
					}

					const double radial_angle = std::atan2(hit_it->second.second, hit_it->second.first);
					const double segment_angle = std::atan2(segments[i].dy, segments[i].dx);
					if (angular_distance(segment_angle, radial_angle) > cfg.layer01_radial_tolerance) {
						h[i] -= cfg.layer_radius_penalty;
					}
				}
			}

		// Run annealing on the Ising expansion of the binary segment-selection objective.
		AnnealingResult result = main_simulation(N,
												 J,
												 h,
												 cfg.t_min,
												 cfg.t_max,
												 cfg.n_steps,
												 cfg.toll,
												 cfg.eq_sweeps,
												 cfg.seed,
												 cfg.log_every_steps,
												 cfg.checkpoint_every_steps);

		const std::filesystem::path out_dir(cfg.out_dir);
		std::filesystem::create_directories(out_dir);

		// Persist final spin assignment and run metadata for downstream analysis.
		write_final_state_csv(out_dir / "final_state.csv", result.state);
		write_lowest_energy_state_csv(out_dir / "lowest_energy_state.csv", result.best_state);
		write_energy_trace_csv(out_dir / "energy_trace.csv", result.trace);
		write_state_checkpoints_csv(out_dir / "state_checkpoints.csv", result.checkpoints, N);
		write_meta_json(
			out_dir / "annealing_meta.json",
			cfg,
			N,
			count_undirected_edges(J),
			static_cast<int>(result.trace.size()),
			static_cast<int>(result.checkpoints.size()),
			result.best_energy
		);

		std::cout << "Saved annealing artifacts to: " << out_dir << "\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}
}
