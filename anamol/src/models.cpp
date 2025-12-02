#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cassert>

#include "ext.h"
#include "instr.h"
#include "models.h"
#include "params.h"
#include "resources.h"

namespace analytical {

using std::vector;

const uint64_t CACHE_LINE_SIZE = 64;
const uint64_t LARGE_CONSTANT = 1000000000ULL;

////////////////////////////////////////////////////////////////////////////
// Base calculations
////////////////////////////////////////////////////////////////////////////

/* Implements Algorithm 1 from the paper. */
uint64_t resp_cycle(
  uint64_t req_cycle,                           // request cycle for the instruction
  Instr& instr,                                  // instruction
  std::map<uint64_t, uint64_t>& last_req_cycles, // last request cycles for each cache line
  std::map<uint64_t, uint64_t>& last_resp_cycles // last response cycles for each cache line
) {
  uint64_t resp_cycle;
  if (instr.is_load) {
    /* Improve on the in-order cache simulation's memory model 
       for load instructions. */
    assert(instr.read_address != 0); // load addresses must have a read address
    uint64_t cache_line = instr.read_address / CACHE_LINE_SIZE;
    auto it = last_req_cycles.find(cache_line);
    if (it == last_req_cycles.end()) {
      /* If there isn't already an entry for the cache line,
         we need to make a request, so set entries for the
         cache line in all state variables. */
      last_req_cycles[cache_line] = 0;
      last_resp_cycles[cache_line] = 0;
    }
    /* req_cycle must be non-decreasing for requests
        for the same cache line. */
    assert(req_cycle >= last_req_cycles[cache_line]);
    uint64_t prev_resp_cycle = last_resp_cycles[cache_line];
    resp_cycle = std::max(req_cycle + instr.exe_latency, prev_resp_cycle);
    
    // update state variables
    last_resp_cycles[cache_line] = resp_cycle;
    last_req_cycles[cache_line] = req_cycle;
  } else {
    /* Constant latency for non-load instructions. */
    resp_cycle = instr.exe_latency + req_cycle;
  }

  return resp_cycle;
}

// 1. ROB (Reorder Buffer) Throughput
double get_thr_rob(const vector<Instr>& window, uint16_t rob_size) {
  instr_id_t k = window.size();
  instr_id_t firstID = window[0].id;

  vector<unsigned> arrival(k);      // a_i
  vector<unsigned> start_cycle(k);  // s_i
  vector<unsigned> finish(k);       // f_i
  vector<unsigned> commit(k);       // c_i

  std::map<uint64_t, uint64_t> last_req_cycles;
  std::map<uint64_t, uint64_t> last_resp_cycles;

  for (unsigned i = 0; i < k; ++i) {
    const auto& instr = window[i];

    // a_i = c_{i-ROB}
    // For first ROB instructions, arrival time is 0
    if (i < rob_size) {
      arrival[i] = 0;
    } else {
      arrival[i] = commit[i - rob_size];
    }

    // s_i = max(a_i, max{f_d | d in Dep(i)})
    instr_id_t max_dep_finish = arrival[i];
    for (instr_id_t dep_id : instr.deps) {
      int dep_idx = dep_id - firstID;
      if (dep_idx >= 0 && dep_idx < (int)i) {
        // Dependency is in the current window and already processed
        max_dep_finish = std::max(max_dep_finish, finish[dep_idx]);
      }
      // Dependencies outside the window or from future instructions are ignored
      // (assumes they're either complete or will be handled by dependency
      // tracking)
    }
    start_cycle[i] = max_dep_finish;

    // f_i = RespCycle(s_i, instr_i)
    finish[i] = resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

    // c_i = max(f_i, c_{i-1})
    // Instructions must commit in order
    if (i == 0) {
      commit[i] = finish[i];
    } else {
      commit[i] = std::max(finish[i], commit[i - 1]);
    }
  }

  // throughput = k / c_{k-1}
  // Total cycles is the commit time of the last instruction
  uint32_t total_cycles = commit[k - 1];

  // Edge case: if all instructions complete at cycle 0 (very unlikely)
  // Return a very high throughput rather than dividing by zero
  if (total_cycles == 0) {
    return k;
  }

  return (double)k / total_cycles;
}

// 2. Load Queue Throughput
double get_thr_load_queue(const vector<Instr>& window,
                          uint16_t load_queue_size) {
  // Filter to only load instructions
  vector<const Instr*> loads;
  for (const auto& instr : window) {
    if (instr.is_load) {
      loads.push_back(&instr);
    }
  }

  // If no loads, this resource is not a bottleneck
  if (loads.empty()) return window.size();

  uint32_t k = window.size();
  uint32_t n_loads = loads.size();

  vector<unsigned> arrival(n_loads);      // a_i
  vector<unsigned> start_cycle(n_loads);  // s_i
  vector<unsigned> finish(n_loads);       // f_i
  vector<unsigned> commit(n_loads);       // c_i

  std::map<uint64_t, uint64_t> last_req_cycles;
  std::map<uint64_t, uint64_t> last_resp_cycles;

  for (int i = 0; i < (int)n_loads; ++i) {
    const auto& instr = *loads[i];

    // a_i = c_{i-LQ}
    // For first LQ loads, arrival time is 0
    if (i < (int)load_queue_size) {
      arrival[i] = 0;
    } else {
      arrival[i] = commit[i - load_queue_size];
    }

    // s_i = a_i (no dependency constraints for Load Queue model)
    start_cycle[i] = arrival[i];

    // f_i = RespCycle(s_i, instr_i)
    finish[i] = resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

    // c_i = max(f_i, c_{i-1})
    // Loads must commit in program order
    if (i == 0) {
      commit[i] = finish[i];
    } else {
      commit[i] = std::max(finish[i], commit[i - 1]);
    }
  }

  // throughput = k / c_{n_loads-1}
  // Total cycles based on last load's commit time
  uint32_t total_cycles = commit[n_loads - 1];

  // Edge case: if all loads complete at cycle 0
  if (total_cycles == 0) return window.size();

  return (double)k / total_cycles;
}

// 3. Store Queue Throughput
double get_thr_store_queue(const vector<Instr>& window,
                           uint16_t store_queue_size) {
  // Filter to only store instructions
  vector<const Instr*> stores;
  for (const auto& instr : window) {
    if (instr.is_store) {
      stores.push_back(&instr);
    }
  }

  // If no stores, this resource is not a bottleneck
  if (stores.empty()) return window.size();

  uint32_t k = window.size();
  uint32_t n_stores = stores.size();

  vector<unsigned> arrival(n_stores);      // a_i
  vector<unsigned> start_cycle(n_stores);  // s_i
  vector<unsigned> finish(n_stores);       // f_i
  vector<unsigned> commit(n_stores);       // c_i

  std::map<uint64_t, uint64_t> last_req_cycles;
  std::map<uint64_t, uint64_t> last_resp_cycles;

  for (int i = 0; i < (int)n_stores; ++i) {
    const auto& instr = *stores[i];

    // a_i = c_{i-SQ}
    // For first SQ stores, arrival time is 0
    if (i < (int)store_queue_size) {
      arrival[i] = 0;
    } else {
      arrival[i] = commit[i - store_queue_size];
    }

    // s_i = a_i (no dependency constraints for Store Queue model)
    start_cycle[i] = arrival[i];

    // f_i = RespCycle(s_i, instr_i)
    finish[i] = resp_cycle(start_cycle[i], instr, last_req_cycles, last_resp_cycles);

    // c_i = max(f_i, c_{i-1})
    // Stores must commit in program order
    if (i == 0) {
      commit[i] = finish[i];
    } else {
      commit[i] = std::max(finish[i], commit[i - 1]);
    }
  }

  // throughput = k / c_{n_stores-1}
  // Total cycles based on last store's commit time
  uint32_t total_cycles = commit[n_stores - 1];

  // Edge case: if all stores complete at cycle 0
  if (total_cycles == 0) return window.size();

  return (double)k / total_cycles;
}

// 4. ALU Issue Width Throughput
double get_thr_alu_issue(const vector<Instr>& window,
                         uint16_t alu_issue_width) {
  // Count ALU instructions in the window
  int n_alu = 0;
  for (const auto& instr : window) {
    n_alu += instr.is_alu;
  }

  // Handle edge case: no ALU instructions
  if (n_alu == 0) return window.size();

  // throughput = k / (n_ALU / alu_issue_width)
  uint32_t k = window.size();
  double cycles_needed = (double)n_alu / alu_issue_width;
  return k / cycles_needed;
}

// 5. Floating-Point Issue Width Throughput
double get_thr_fp_issue(const vector<Instr>& window, uint16_t fp_issue_width) {
  // Count FP instructions in the window
  int n_fp = 0;
  for (const auto& instr : window) {
    n_fp += instr.is_fp;
  }

  // Handle edge case: no FP instructions
  if (n_fp == 0) return window.size();

  // throughput = k / (n_FP / fp_issue_width)
  uint32_t k = window.size();
  double cycles_needed = (double)n_fp / fp_issue_width;
  return k / cycles_needed;
}

// 6. Load-Store Issue Width Throughput
double get_thr_ls_issue(const vector<Instr>& window, uint16_t ls_issue_width) {
  // Count Load/Store instructions in the window
  int n_ls = 0;
  for (const auto& instr : window) {
    n_ls += instr.is_load + instr.is_store;
  }

  // Handle edge case: no Load/Store instructions
  if (n_ls == 0) return window.size();

  // throughput = k / (n_LS / ls_issue_width)
  uint32_t k = window.size();
  double cycles_needed = (double)n_ls / ls_issue_width;
  return k / cycles_needed;
}

// 7. Load/Load-Store Pipes Lower Bound Throughput
double get_thr_ls_pipes_lower(const vector<Instr>& window,
                              uint16_t num_ls_pipes, uint16_t num_load_pipes) {
  // Based on worst-case allocation scenario
  int n_load = 0, n_store = 0;
  for (const auto& instr : window) {
    n_load += instr.is_load;
    n_store += instr.is_store;
  }

  if (n_load == 0 && n_store == 0) return window.size();

  double t_max = (double)n_load / (num_ls_pipes + num_load_pipes) +
                 (double)n_store / num_ls_pipes;

  if (t_max == 0) return window.size();

  return window.size() / t_max;
}

// 8. Load/Load-Store Pipes Upper Bound Throughput
double get_thr_ls_pipes_upper(const vector<Instr>& window,
                              uint16_t num_ls_pipes, uint16_t num_load_pipes) {
  // Based on best-case allocation scenario
  int n_load = 0, n_store = 0;
  for (const auto& instr : window) {
    n_load += instr.is_load;
    n_store += instr.is_store;
  }

  if (n_load == 0 && n_store == 0) return window.size();

  // Best-case: stores use LSPs while loads use LPs concurrently
  // Time to complete all stores using LS pipes
  double t_store = (double)n_store / num_ls_pipes;

  // During t_store cycles, load-only pipes can process loads
  double loads_via_load_pipes = t_store * num_load_pipes;

  // Remaining loads after concurrent execution
  double n_remaining_load = std::max(0.0, n_load - loads_via_load_pipes);

  // Remaining loads share LS pipes after stores complete
  double t_min = t_store + n_remaining_load / (num_ls_pipes + num_load_pipes);

  if (t_min == 0) return window.size();

  return window.size() / t_min;
}

// 9. I-Cache Fills Throughput
double get_thr_icache_fills(const vector<Instr>& window, uint16_t max_icache_fills) {
  if (window.empty()) {
    return 0.0;
  }
  
  /* Stores the state related to in-flight requests. Keys are the
     cache lines, values are the cycles at which the requests
     complete. */
  std::map<uint64_t, uint64_t> in_flight_requests;

  /* Cycle at which the previous instruction was issued. For this
     simulation, we assume in-order issue. */
  uint64_t prev_inst_ready_cycle = 0;

  for (const auto& instr : window) {
    uint64_t cache_line = instr.IP / CACHE_LINE_SIZE;
    uint64_t fill_latency = instr.mem_latency;

    auto it = in_flight_requests.find(cache_line);

    if (it != in_flight_requests.end()) {
      /* Request for this instruction's cache line is already in flight. */
      uint64_t completion_cycle = it->second;
      
      /* Enforce in-order constraint: even if the cache line arrives, the instruction
         isn't ready until the previous one is. */
      prev_inst_ready_cycle = std::max(prev_inst_ready_cycle, completion_cycle);
    } else {
      /* We need to issue an icache request for this instruction's cache line. */
      uint64_t next_available_slot_cycle = prev_inst_ready_cycle;

      while (in_flight_requests.size() >= max_icache_fills) {
        uint64_t earliest_finish_time = LARGE_CONSTANT;
        uint64_t earliest_cache_line = 0;

        for (const auto& pair : in_flight_requests) {
          if (pair.second < earliest_finish_time) {
            earliest_finish_time = pair.second;
            earliest_cache_line = pair.first;
          }
        }

        next_available_slot_cycle = earliest_finish_time;
        in_flight_requests.erase(earliest_cache_line);
      }

      uint64_t request_start_cycle = std::max(prev_inst_ready_cycle, next_available_slot_cycle);
      uint64_t completion_cycle = request_start_cycle + fill_latency;

      in_flight_requests[cache_line] = completion_cycle;
      prev_inst_ready_cycle = completion_cycle;
    }
  }

  uint64_t total_cycles = prev_inst_ready_cycle;
  size_t total_instructions = window.size();

  if (total_cycles == 0) {
    return static_cast<double>(total_instructions);
  }

  return static_cast<double>(total_instructions) / static_cast<double>(total_cycles);
}

// 10. Fetch Buffers Throughput
double get_thr_fetch_buffers(const vector<Instr>& window,
                             uint16_t num_fetch_buffers) {
  // TODO: Implement Fetch Buffers throughput via discrete-event simulation
  // Model fetch buffer occupancy constraints
  return 0.0;
}

// main entry
PerResThrVecs get_throughput(vector<Instr> instr_trace, int window_size) {
  if (instr_trace.empty()) return {};

  PerResThrVecs PER_RES_THR_VECS;

  // clear any previous results (single-thread)
  for (auto& v : PER_RES_THR_VECS) v.clear();

  size_t num_windows = (instr_trace.size() + window_size - 1) / window_size;

  // Parallel over resources, parameter sweeps, and windows
#pragma omp parallel
  {
    // We will build thread-local results, then merge
    PerResThrVecs local_thr_vecs;

    // Iterate resources in parallel (all threads share work)
#pragma omp for schedule(dynamic)
    for (size_t res_idx = 0; res_idx < (size_t)Resource::COUNT; ++res_idx) {
      const auto& res_meta = RESOURCE_TABLE[res_idx];
      auto& local_vecs_for_res = local_thr_vecs[res_idx];

      // Pre-size local_vecs_for_res to number of param combos
      size_t num_param_combos = std::distance(res_meta.param_sweep.begin(),
                                              res_meta.param_sweep.end());
      local_vecs_for_res.resize(num_param_combos);

      // Parallelize over parameter combinations
      size_t combo_idx = 0;
      for (const auto& params : res_meta.param_sweep) {
        size_t p_idx = combo_idx++;

        ThrVec tv;
        tv.double_params = (params.size() > 1);
        tv.p0 = params.size() > 0 ? params[0] : 0;
        tv.p1 = params.size() > 1 ? params[1] : 0;
        tv.data.resize(num_windows);

        // Parallelize over windows
#pragma omp parallel for schedule(static)
        for (int win_idx = 0; win_idx < (int)num_windows; ++win_idx) {
          size_t start_idx = (size_t)win_idx * window_size;
          size_t end_idx =
              std::min(start_idx + (size_t)window_size, instr_trace.size());

          vector<Instr> window(instr_trace.begin() + start_idx,
                               instr_trace.begin() + end_idx);

          double thr = res_meta.get_thr(window, params);
          tv.data[win_idx] = thr;
        }

        local_vecs_for_res[p_idx] = std::move(tv);
      }
    }

    // Merge thread-local results into global PER_RES_THR_VECS
#pragma omp critical
    {
      for (size_t res_idx = 0; res_idx < (size_t)Resource::COUNT; ++res_idx) {
        auto& global_vecs = PER_RES_THR_VECS[res_idx];
        auto& local_vecs = local_thr_vecs[res_idx];
        if (!local_vecs.empty()) {
          // Initialize global container once
          if (global_vecs.empty()) {
            global_vecs.resize(local_vecs.size());
          }
          // Merge each ThrVec (same param index goes to same slot)
          for (size_t i = 0; i < local_vecs.size(); ++i) {
            if (!local_vecs[i].data.empty()) {
              // If this slot is unused, move it
              if (global_vecs[i].data.empty()) {
                global_vecs[i] = std::move(local_vecs[i]);
              }
            }
          }
        }
      }
    }
  }  // end outer omp parallel

  return PER_RES_THR_VECS;
}

// Helper: map Resource enum to file-friendly name
static const char* resource_file_name(Resource res) {
  switch (res) {
    case Resource::ROB:
      return "ROB";
    case Resource::LOAD_QUEUE:
      return "LOAD_QUEUE";
    case Resource::STORE_QUEUE:
      return "STORE_QUEUE";
    case Resource::ALU_ISSUE:
      return "ALU_ISSUE";
    case Resource::FP_ISSUE:
      return "FP_ISSUE";
    case Resource::LS_ISSUE:
      return "LS_ISSUE";
    case Resource::LOAD_LS_PIPES_LOWER:
      return "LOAD_LS_PIPES_LOWER";
    case Resource::LOAD_LS_PIPES_UPPER:
      return "LOAD_LS_PIPES_UPPER";
    case Resource::ICACHE_FILLS:
      return "ICACHE_FILLS";
    case Resource::FETCH_BUFFERS:
      return "FETCH_BUFFERS";
    default:
      return "UNKNOWN";
  }
}

// Minimal NumPy .npy v1.0 writer for 2D float64 arrays in C-order.
// shape: (rows, cols), data: rows*cols doubles, row-major.
static void write_npy_2d_float64(const std::string& filename, size_t rows,
                                 size_t cols, const std::vector<double>& data) {
  // Magic string: \x93NUMPY
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  const uint8_t major = 1;
  const uint8_t minor = 0;

  // Build the header dict as a Python literal string.
  // We use little-endian float64: "<f8", Fortran order: False, and given shape.
  std::ostringstream header_ss;
  header_ss << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << rows;
  if (cols == 1) {
    header_ss << ",)";  // 1D shape (rows,)
  } else {
    header_ss << ", " << cols << "),";
  }
  header_ss << "}";

  std::string header = header_ss.str();

  // Pad with spaces and newline so that (magic + 2 + 2 + header) % 64 == 0.
  // Layout: magic(6) + major(1) + minor(1) + header_len_le(2) + header_bytes
  const size_t header_len = header.size();
  // Compute total header + prefix length (not including magic+version+len
  // field) In v1.0 the 2-byte little-endian length is just the header length.
  size_t pre_header = sizeof(magic) + 2 /*major+minor*/ + 2 /*len field*/;
  size_t total = pre_header + header_len;
  size_t pad = 64 - (total % 64);
  if (pad == 64) pad = 0;

  header.append(pad, ' ');
  // Last char must be newline
  if (header.empty() || header.back() != '\n') {
    if (!header.empty())
      header.back() = '\n';
    else
      header.push_back('\n');
  }

  uint16_t header_len_le = static_cast<uint16_t>(header.size());

  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  ofs.write(reinterpret_cast<const char*>(magic), sizeof(magic));
  ofs.put(static_cast<char>(major));
  ofs.put(static_cast<char>(minor));
  ofs.write(reinterpret_cast<const char*>(&header_len_le),
            sizeof(header_len_le));
  ofs.write(header.data(), header.size());

  // Data is already in row-major order.
  ofs.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(double)));
  ofs.close();
}

void export_throughputs(PerResThrVecs PER_RES_THR_VECS) {
  const size_t num_resources = static_cast<size_t>(Resource::COUNT);

  // Ensure output directory exists (simple, portable approach via system()).
  std::system("mkdir -p output");

  for (size_t r = 0; r < num_resources; ++r) {
    Resource res = static_cast<Resource>(r);
    const auto& thr_vecs = PER_RES_THR_VECS[r];

    // Skip resources with no data
    if (thr_vecs.empty()) continue;

    // Infer number of windows from first ThrVec
    const size_t num_windows = thr_vecs[0].data.size();
    if (num_windows == 0) continue;

    const bool double_params = thr_vecs[0].double_params;
    // col 0: param count (1 or 2)
    // col 1: p0
    // col 2: p1 (if double_params)
    const size_t param_cols = double_params ? 3 : 2;
    const size_t rows = thr_vecs.size();
    const size_t cols = param_cols + num_windows;

    std::vector<double> array(rows * cols, 0.0);

    for (size_t i = 0; i < rows; ++i) {
      const ThrVec& tv = thr_vecs[i];
      size_t base = i * cols;

      // Param count
      array[base + 0] = double_params ? 2.0 : 1.0;

      // Parameter values
      array[base + 1] = static_cast<double>(tv.p0);
      if (double_params) {
        array[base + 2] = static_cast<double>(tv.p1);
      }

      // Throughput data starts after param_cols
      const size_t w = std::min(num_windows, tv.data.size());
      for (size_t j = 0; j < w; ++j) {
        array[base + param_cols + j] = tv.data[j];
      }
    }

    // Build lowercase filename: output/thr_<resource>.npy
    std::string res_name = resource_file_name(res);
    for (auto& c : res_name) c = static_cast<char>(std::tolower(c));

    std::string fname = std::string("output/thr_") + res_name + ".npy";
    write_npy_2d_float64(fname, rows, cols, array);
  }
}

}  // namespace analytical