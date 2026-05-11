#include <getopt.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "instr.h"
#include "models.h"
#include "npy_reader.h"
#include "parser.h"

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " [-w WINDOW_SIZE|--window WINDOW_SIZE] "
               "[-t TRACE|--tracefile TRACE] "
               "[-o OUTPUT_DIR|--output-dir OUTPUT_DIR] "
               "[-l LATENCIES_NPY|--latencies-npy LATENCIES_NPY] "
               "[-c CONFIG_JSON|--config-json CONFIG_JSON]\n"
               "Defaults: WINDOW_SIZE=400, TRACE=trace.csv\n"
               "          OUTPUT_DIR=output/<trace_stem>\n"
               "\n"
               "Without --latencies-npy: single run using latencies from TRACE.\n"
               "With    --latencies-npy: runs latency-independent resources once,\n"
               "  then iterates over every cache config in the .npy, writing\n"
               "  latency-dependent results to OUTPUT_DIR/config_NNNN/.\n"
               "\n"
               "Without --config-json: sweeps all param combinations (default).\n"
               "With    --config-json: computes only for the specified config.\n"
               "  JSON format: {\"rob_size\": 128, \"load_queue_size\": 64, ...}\n";
}

// Minimal JSON parser for a flat {string: integer} object.
static std::map<std::string, uint16_t> parse_config_json(
    const std::string& json) {
  std::map<std::string, uint16_t> config;
  std::string s = json;

  // Strip whitespace
  s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());

  if (s.empty() || s[0] != '{' || s.back() != '}')
    throw std::runtime_error("config-json must be a JSON object {...}");
  s = s.substr(1, s.size() - 2);

  if (s.empty()) return config;

  std::istringstream iss(s);
  std::string token;
  while (std::getline(iss, token, ',')) {
    if (token.empty()) continue;
    auto colon = token.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("Malformed JSON pair (no ':'): " + token);

    std::string key = token.substr(0, colon);
    std::string val_str = token.substr(colon + 1);

    // Remove surrounding quotes from key
    auto q1 = key.find('"');
    auto q2 = key.rfind('"');
    if (q1 != std::string::npos && q2 != q1)
      key = key.substr(q1 + 1, q2 - q1 - 1);

    config[key] = static_cast<uint16_t>(std::stoi(val_str));
  }
  return config;
}

// Return the filename stem (no directory, no extension) of a path.
// e.g. "traces/collatz_trace_with_latency.csv" -> "collatz_trace_with_latency"
static std::string stem_of(const std::string& path) {
  std::string base = path;
  auto slash = base.rfind('/');
  if (slash != std::string::npos) base = base.substr(slash + 1);
  auto dot = base.rfind('.');
  if (dot != std::string::npos) base = base.substr(0, dot);
  return base;
}

static std::string zero_pad(size_t n, int width) {
  std::ostringstream ss;
  ss << std::setfill('0') << std::setw(width) << n;
  return ss.str();
}

int main(int argc, char* argv[]) {
  std::string csv_file = "trace.csv";
  int window_size = 400;
  std::string output_dir;       // empty = auto-derive from csv_file
  std::string latencies_npy;    // empty = single-run mode
  std::string config_json;      // empty = full sweep (default)

  const char* short_opts = "hw:t:o:l:c:";
  const option long_opts[] = {
      {"help",          no_argument,       nullptr, 'h'},
      {"window",        required_argument, nullptr, 'w'},
      {"tracefile",     required_argument, nullptr, 't'},
      {"output-dir",    required_argument, nullptr, 'o'},
      {"latencies-npy", required_argument, nullptr, 'l'},
      {"config-json",   required_argument, nullptr, 'c'},
      {nullptr, 0, nullptr, 0},
  };

  while (true) {
    int opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (opt == -1) break;

    switch (opt) {
      case 'h':
        print_usage(argv[0]);
        return 0;
      case 'w':
        window_size = std::stoi(optarg);
        break;
      case 't':
        csv_file = optarg;
        break;
      case 'o':
        output_dir = optarg;
        break;
      case 'l':
        latencies_npy = optarg;
        break;
      case 'c':
        config_json = optarg;
        break;
      case '?':
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  // Derive output directory from trace stem if not provided.
  if (output_dir.empty()) {
    output_dir = "output/" + stem_of(csv_file);
  }

  std::cout << "Analytical model driver\n";
  std::cout << "Trace file : " << csv_file << "\n";
  std::cout << "Window size: " << window_size << "\n";
  std::cout << "Output dir : " << output_dir << "\n";
  if (!latencies_npy.empty())
    std::cout << "Latencies  : " << latencies_npy << "\n";
  if (!config_json.empty())
    std::cout << "Config     : " << config_json << " (single-config mode)\n";

  // Parse single config if provided
  bool single_config_mode = !config_json.empty();
  std::map<std::string, uint16_t> single_config;
  if (single_config_mode) {
    try {
      single_config = parse_config_json(config_json);
    } catch (const std::exception& e) {
      std::cerr << "Error parsing --config-json: " << e.what() << "\n";
      return 1;
    }
  }

  // ROB latency analysis is always run for the default 11-size sweep
  // ({1,2,4,...,1024}). Downstream models consume per-size features
  // (rob1_issue_*, rob2_issue_*, ...); narrowing to the config's rob_size
  // would break that feature set. Leave this empty so the C++ side falls
  // back to its built-in default list.
  std::vector<uint16_t> rob_sizes_for_analysis;

  std::cout << "\nParsing and converting trace...\n";
  std::vector<analytical::Instr> instrs =
      analytical::parse_and_convert(csv_file);
  std::cout << "Converted " << instrs.size() << " instructions\n";

  if (instrs.empty()) {
    std::cerr << "Error: No instructions parsed from " << csv_file << "\n";
    return 1;
  }

  std::size_t num_windows =
      (instrs.size() + static_cast<std::size_t>(window_size) - 1) /
      static_cast<std::size_t>(window_size);
  std::cout << "Number of windows: " << num_windows << "\n";

  if (latencies_npy.empty()) {
    // ── Single-run mode ──────────────────────────────────────────────────────
    std::cout << "\nCalculating throughput...\n";
    analytical::PerResThrVecs per_res_thr_vecs =
        single_config_mode
            ? analytical::get_throughput_single_config(instrs, window_size,
                                                        single_config)
            : analytical::get_throughput(instrs, window_size);

    std::cout << "Exporting throughputs to " << output_dir << " ...\n";
    analytical::export_throughputs(per_res_thr_vecs, output_dir);
    std::cout << "Done.\n";

    std::cout << "\nCalculating ROB latency analysis...\n";
    std::vector<analytical::RobLatencyData> latency_data =
        analytical::get_rob_latency_analysis(instrs, rob_sizes_for_analysis);

    std::cout << "\nExporting latency analysis to " << output_dir << " ...\n";
    analytical::export_latency_analysis(latency_data, output_dir);
    std::cout << "Done.\n";

  } else {
    // ── Per-cache-config mode ────────────────────────────────────────────────
    auto npy = analytical::open_npy_latencies(latencies_npy, instrs.size());
    std::cout << "Loaded latencies: " << npy.n_configs << " configs × "
              << npy.n_instrs << " instrs\n";

    // Latency-independent resources: compute once with the trace's own latencies
    std::cout << "\nCalculating latency-independent throughputs...\n";
    analytical::PerResThrVecs thr_indep =
        single_config_mode
            ? analytical::get_throughput_single_config(instrs, window_size,
                                                        single_config, false)
            : analytical::get_throughput(instrs, window_size, false);
    analytical::export_throughputs(thr_indep, output_dir);
    std::cout << "Exported to " << output_dir << "\n";

    // Latency-dependent resources + ROB latency analysis: once per cache config
    std::cout << "\nRunning " << npy.n_configs
              << " cache configs (latency-dependent resources + ROB analysis)...\n";

    for (size_t cfg = 0; cfg < npy.n_configs; ++cfg) {
      auto slice = npy.slice(cfg);
      for (size_t i = 0; i < instrs.size(); ++i) {
        instrs[i].fetch_latency = slice[i * 2];
        instrs[i].exe_latency   = slice[i * 2 + 1];
      }

      std::string cfg_dir = output_dir + "/config_" + zero_pad(cfg, 4);

      analytical::PerResThrVecs thr_dep =
          single_config_mode
              ? analytical::get_throughput_single_config(instrs, window_size,
                                                          single_config, true)
              : analytical::get_throughput(instrs, window_size, true);
      analytical::export_throughputs(thr_dep, cfg_dir);

      std::vector<analytical::RobLatencyData> lat =
          analytical::get_rob_latency_analysis(instrs, rob_sizes_for_analysis);
      analytical::export_latency_analysis(lat, cfg_dir);

      std::cout << "  [" << (cfg + 1) << "/" << npy.n_configs << "] config_"
                << zero_pad(cfg, 4) << "\n";
    }

    std::cout << "Done.\n";
  }

  return 0;
}
