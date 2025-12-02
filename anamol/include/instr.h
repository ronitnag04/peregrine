#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <string>
#include <vector>

using latency_t = uint16_t;

namespace tracing {

struct mem_access_t {
  unsigned long addr;
  unsigned int size;
};

struct instr_trace_t {
  unsigned long ip;
  std::string assembly;
  std::string category;
  std::string opcode;
  std::string branch_type;
  bool branch_taken;
  unsigned long branch_target_addr;
  bool inst_sync;
  std::vector<std::string> read_registers;
  std::vector<std::string> write_registers;
  std::vector<unsigned long> reg_dependent_ips;
  std::vector<mem_access_t> read_addresses;
  std::vector<mem_access_t> write_addresses;
  std::vector<unsigned long> mem_dependent_ips;
  latency_t exe_latency;
  latency_t fetch_latency;
};

}  // namespace tracing

namespace analytical {

using instr_id_t = uint32_t;

enum class branch_t : uint8_t { DIRECT_COND, DIRECT_UNCOND, INDIRECT };

struct Instr {
  uint64_t IP;
  instr_id_t id;
  latency_t exe_latency;
  latency_t mem_latency;

  bool is_alu;
  bool is_fp;
  bool is_load;
  bool is_store;
  bool is_isb;
  bool is_branch;

  branch_t branch_type;
  std::vector<uint32_t> deps;
  uint64_t read_address;
};

}  // namespace analytical

#endif  // TYPES_H