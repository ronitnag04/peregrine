#include <unordered_map>

#include "convert_trace.h"

namespace analytical {

std::vector<Instr> convert_trace(
    const std::vector<tracing::instr_trace_t>& trace_data) {
  std::vector<Instr> instructions;
  instructions.reserve(trace_data.size());

  // Map IP addresses to instruction IDs for dependency tracking
  std::unordered_map<uint64_t, instr_id_t> ip_to_id;

  for (size_t i = 0; i < trace_data.size(); i++) {
    const auto& src = trace_data[i];
    Instr instr;

    // Basic fields
    instr.IP = src.ip;
    instr.id = static_cast<instr_id_t>(i);
    ip_to_id[src.ip] = instr.id;

    // Default latency (can be refined based on category/opcode)
    instr.latency = 1;

    // Categorize instruction type
    instr.is_alu = (src.category.find("ALU") != std::string::npos ||
                    src.category.find("LOGICAL") != std::string::npos ||
                    src.category.find("SHIFT") != std::string::npos);

    instr.is_fp = (src.category.find("FP") != std::string::npos ||
                   src.category.find("SIMD") != std::string::npos ||
                   src.category.find("SSE") != std::string::npos ||
                   src.category.find("AVX") != std::string::npos);

    instr.is_load = !src.read_addresses.empty();
    instr.is_store = !src.write_addresses.empty();

    instr.is_isb = src.inst_sync;

    instr.is_branch = !src.branch_type.empty();

    // Parse branch type
    if (instr.is_branch) {
      if (src.branch_type.find("DIRECT_COND") != std::string::npos) {
        instr.branch_type = branch_t::DIRECT_COND;
      } else if (src.branch_type.find("DIRECT_UNCOND") != std::string::npos) {
        instr.branch_type = branch_t::DIRECT_UNCOND;
      } else if (src.branch_type.find("INDIRECT") != std::string::npos) {
        instr.branch_type = branch_t::INDIRECT;
      } else {
        instr.branch_type = branch_t::DIRECT_COND;  // default
      }
    } else {
      instr.branch_type = branch_t::DIRECT_COND;  // placeholder
    }

    // Convert dependencies from IPs to instruction IDs
    // Combine both register and memory dependencies
    std::vector<uint64_t> all_dep_ips;
    all_dep_ips.insert(all_dep_ips.end(), src.reg_dependent_ips.begin(),
                       src.reg_dependent_ips.end());
    all_dep_ips.insert(all_dep_ips.end(), src.mem_dependent_ips.begin(),
                       src.mem_dependent_ips.end());

    for (uint64_t dep_ip : all_dep_ips) {
      auto it = ip_to_id.find(dep_ip);
      if (it != ip_to_id.end()) {
        instr.deps.push_back(it->second);
      }
    }

    // Calculate I-cache line (assuming 64-byte cache lines)
    instr.icache_line = src.ip / 64;

    instructions.push_back(instr);
  }

  return instructions;
}

std::vector<Instr> read_trace(const std::string& filename) {
  // Parse the CSV file
  // Convert to analytical instructions
  return analytical::convert_trace(tracing::parse_trace_csv(filename));
}

}  // namespace analytical