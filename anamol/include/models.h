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
// Base Throughput Calculations (declarations auto-generated from registry.yaml)
////////////////////////////////////////////////////////////////////////////
#include "models_decl_gen.h"

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
