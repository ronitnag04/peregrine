#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace analytical {

// Minimal reader for the (N_configs, N_instrs, 2) uint16 .npy files produced
// by annotate_trace.py.  Only supports numpy v1.0, C-order, dtype uint16.
//
// Usage:
//   auto npy = open_npy_latencies(path, instrs.size());
//   for (size_t cfg = 0; cfg < npy.n_configs; ++cfg) {
//       auto s = npy.slice(cfg);   // s[i*2] = fetch_lat, s[i*2+1] = exe_lat
//       instrs[i].fetch_latency = s[i*2];
//       instrs[i].exe_latency   = s[i*2+1];
//   }

struct NpyLatencies {
  size_t      n_configs;
  size_t      n_instrs;
  std::string path;
  long        data_offset;  // byte position of first data element

  // Return flat [fetch_lat_0, exe_lat_0, fetch_lat_1, exe_lat_1, ...]
  // for the given config index.
  std::vector<uint16_t> slice(size_t config_idx) const {
    if (config_idx >= n_configs)
      throw std::out_of_range("config_idx out of range");

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);

    long byte_pos = data_offset +
                    static_cast<long>(config_idx * n_instrs * 2) *
                        static_cast<long>(sizeof(uint16_t));
    std::fseek(f, byte_pos, SEEK_SET);

    std::vector<uint16_t> result(n_instrs * 2);
    size_t read = std::fread(result.data(), sizeof(uint16_t), n_instrs * 2, f);
    std::fclose(f);

    if (read != n_instrs * 2)
      throw std::runtime_error("Short read from: " + path);
    return result;
  }
};

inline NpyLatencies open_npy_latencies(const std::string& path,
                                       size_t n_instrs) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("Cannot open latencies npy: " + path);

  // numpy v1.0 header layout:
  //   bytes 0-5  : magic \x93NUMPY
  //   byte  6    : major version (1)
  //   byte  7    : minor version (0)
  //   bytes 8-9  : header_len (uint16 little-endian)
  //   bytes 10 .. 10+header_len-1 : Python dict string
  //   bytes 10+header_len ..      : raw data (C-order)
  uint8_t  magic[6];
  uint8_t  major, minor;
  uint16_t header_len_le;

  if (std::fread(magic, 1, 6, f) != 6 ||
      std::fread(&major, 1, 1, f) != 1 ||
      std::fread(&minor, 1, 1, f) != 1 ||
      std::fread(&header_len_le, 2, 1, f) != 1) {
    std::fclose(f);
    throw std::runtime_error("Failed to read npy header: " + path);
  }

  // header_len_le is already little-endian; on LE hosts this is a no-op.
  long data_offset = 10L + static_cast<long>(header_len_le);

  // Determine n_configs from file size
  std::fseek(f, 0, SEEK_END);
  long file_size = std::ftell(f);
  std::fclose(f);

  long data_bytes = file_size - data_offset;
  long bytes_per_config = static_cast<long>(n_instrs) * 2L * sizeof(uint16_t);
  if (bytes_per_config == 0 || data_bytes % bytes_per_config != 0)
    throw std::runtime_error(
        "npy size mismatch — wrong n_instrs or corrupt file: " + path);

  size_t n_configs = static_cast<size_t>(data_bytes / bytes_per_config);
  return {n_configs, n_instrs, path, data_offset};
}

}  // namespace analytical
