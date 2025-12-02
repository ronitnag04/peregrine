#include <getopt.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "instr.h"
#include "models.h"
#include "parser.h"

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " [-w WINDOW_SIZE|--window WINDOW_SIZE] "
               "[-t TRACE|--tracefile TRACE]\n"
               "Defaults: WINDOW_SIZE=400, TRACE=trace.csv\n";
}

int main(int argc, char* argv[]) {
  std::string csv_file = "trace.csv";
  int window_size = 400;

  const char* short_opts = "hw:t:";
  const option long_opts[] = {
      {"help", no_argument, nullptr, 'h'},
      {"window", required_argument, nullptr, 'w'},
      {"tracefile", required_argument, nullptr, 't'},
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
      case '?':
      default:
        print_usage(argv[0]);
        return 1;
    }
  }

  std::cout << "Analytical model driver\n";
  std::cout << "Trace file : " << csv_file << "\n";
  std::cout << "Window size: " << window_size << "\n";

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

  std::cout << "\nCalculating throughput...\n";
  analytical::PerResThrVecs per_res_thr_vecs =
      analytical::get_throughput(instrs, window_size);

  std::cout << "Exporting throughputs to ./output ...\n";
  analytical::export_throughputs(per_res_thr_vecs);
  std::cout << "Done.\n";

  return 0;
}