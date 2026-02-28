#ifndef MODELS_H
#define MODELS_H

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "instr.h"
#include "params_gen.h"    // ParamType, ParamSweep, PARAM_RANGES (generated)
#include "resources_gen.h" // Resource enum (generated)

namespace analytical {

// resp_cycle from Algorithm 1
uint64_t resp_cycle(uint64_t req_cycle, Instr instr,
                    std::map<uint64_t, uint64_t> last_req_cycles,
                    std::map<uint64_t, uint64_t> last_resp_cycles);

////////////////////////////////////////////////////////////////////////////
// Base Throughput Calculations
////////////////////////////////////////////////////////////////////////////
// 1. ROB (Reorder Buffer) Throughput
double get_thr_rob(const std::vector<Instr>& window, uint16_t rob_size);
// 2. Load Queue Throughput
double get_thr_load_queue(const std::vector<Instr>& window,
                          uint16_t load_queue_size);
// 3. Store Queue Throughput
double get_thr_store_queue(const std::vector<Instr>& window,
                           uint16_t store_queue_size);
// 4. ALU Issue Width Throughput
double get_thr_alu_issue(const std::vector<Instr>& window,
                         uint16_t alu_issue_width);
// 4a. ALU Multiply Issue Width Throughput
double get_thr_alu_mul_issue(const std::vector<Instr>& window,
                             uint16_t alu_mul_issue_width);
// 4b. ALU Divide Issue Width Throughput
double get_thr_alu_div_issue(const std::vector<Instr>& window,
                             uint16_t alu_div_issue_width);
// 5. Floating-Point Issue Width Throughput
double get_thr_fp_issue(const std::vector<Instr>& window,
                        uint16_t fp_issue_width);
// 6. Load-Store Issue Width Throughput
double get_thr_ls_issue(const std::vector<Instr>& window,
                        uint16_t ls_issue_width);
// 7. Load/Load-Store Pipes Lower Bound Throughput
double get_thr_load_ls_pipes_lower(const std::vector<Instr>& window,
                                   uint16_t num_ls_pipes,
                                   uint16_t num_load_pipes);
// 8. Load/Load-Store Pipes Upper Bound Throughput
double get_thr_load_ls_pipes_upper(const std::vector<Instr>& window,
                                   uint16_t num_ls_pipes,
                                   uint16_t num_load_pipes);
// 9. I-Cache Fills Throughput
double get_thr_icache_fills(const std::vector<Instr>& window,
                            uint16_t max_icache_fills);
// 10. Fetch Buffers Throughput
double get_thr_fetch_buffers(const std::vector<Instr>& window,
                             uint16_t num_fetch_buffers);

////////////////////////////////////////////////////////////////////////////
// Resource Registry Entry
////////////////////////////////////////////////////////////////////////////
using ThrFunc = std::function<double(const std::vector<Instr>&,
                                     const std::vector<uint16_t>&)>;

// One entry per resource. RESOURCE_REGISTRY (in resource_registry.h, generated)
// holds the full vector. models.cpp includes resource_registry.h directly.
struct ResourceEntry {
  Resource    resource;
  const char* name;     // canonical name — used as .npy filename stem
  bool        enabled;  // false → skip sweep, write no output file
  ThrFunc     func;
  ParamSweep  sweep;
};

////////////////////////////////////////////////////////////////////////////
// Results
////////////////////////////////////////////////////////////////////////////
struct ThrVec {
  std::vector<double> data;  // throughput for each window
  bool double_params = false;
  uint16_t p0 = 0;
  uint16_t p1 = 0;
};

// One vector<ThrVec> per Resource enum value
using PerResThrVecs =
    std::array<std::vector<ThrVec>, static_cast<size_t>(Resource::COUNT)>;

////////////////////////////////////////////////////////////////////////////
// Main Entry
////////////////////////////////////////////////////////////////////////////
PerResThrVecs get_throughput(std::vector<Instr> instr_trace,
                             int window_size = 400);

void export_throughputs(PerResThrVecs PER_RES_THR_VECS);

////////////////////////////////////////////////////////////////////////////
// ROB Latency Analysis
////////////////////////////////////////////////////////////////////////////
struct RobLatencyData {
  uint16_t rob_size;
  double overall_throughput;
  std::vector<uint32_t> issue_latencies;
  std::vector<uint32_t> commit_latencies;
  std::vector<uint32_t> exec_latencies;
};

std::vector<RobLatencyData> get_rob_latency_analysis(
    const std::vector<Instr>& instr_trace);

void export_latency_analysis(const std::vector<RobLatencyData>& latency_data);

}  // namespace analytical

#endif  // MODELS_H
