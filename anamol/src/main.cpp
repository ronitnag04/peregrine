#include <getopt.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
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
               "[-l LATENCIES_NPY|--latencies-npy LATENCIES_NPY]\n"
               "Defaults: WINDOW_SIZE=400, TRACE=trace.csv\n"
               "          OUTPUT_DIR=output/<trace_stem>\n"
               "\n"
               "Without --latencies-npy: single run using latencies from TRACE.\n"
               "With    --latencies-npy: runs latency-independent resources once,\n"
               "  then iterates over every cache config in the .npy, writing\n"
               "  latency-dependent results to OUTPUT_DIR/config_NNNN/.\n";
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

  const char* short_opts = "hw:t:o:l:";
  const option long_opts[] = {
      {"help",          no_argument,       nullptr, 'h'},
      {"window",        required_argument, nullptr, 'w'},
      {"tracefile",     required_argument, nullptr, 't'},
      {"output-dir",    required_argument, nullptr, 'o'},
      {"latencies-npy", required_argument, nullptr, 'l'},
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
    // ── Single-run mode (existing behaviour) ────────────────────────────────
    std::cout << "\nCalculating throughput...\n";
    analytical::PerResThrVecs per_res_thr_vecs =
        analytical::get_throughput(instrs, window_size);

    std::cout << "Exporting throughputs to " << output_dir << " ...\n";
    analytical::export_throughputs(per_res_thr_vecs, output_dir);
    std::cout << "Done.\n";

    std::cout << "\nCalculating ROB latency analysis...\n";
    std::vector<analytical::RobLatencyData> latency_data =
        analytical::get_rob_latency_analysis(instrs);

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
        analytical::get_throughput(instrs, window_size, false);
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
          analytical::get_throughput(instrs, window_size, true);
      analytical::export_throughputs(thr_dep, cfg_dir);

      std::vector<analytical::RobLatencyData> lat =
          analytical::get_rob_latency_analysis(instrs);
      analytical::export_latency_analysis(lat, cfg_dir);

      std::cout << "  [" << (cfg + 1) << "/" << npy.n_configs << "] config_"
                << zero_pad(cfg, 4) << "\n";
    }

    std::cout << "Done.\n";
  }

  return 0;
}
