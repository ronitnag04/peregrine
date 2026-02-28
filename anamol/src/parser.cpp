#include "parser.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "csv.h"
#include "opcode_categories.h"

namespace analytical {

// Helper to parse hex string to unsigned long
unsigned long parse_hex(const std::string& hex_str) {
  if (hex_str.empty()) return 0;
  return std::stoull(hex_str, nullptr, 16);
}

// Helper to parse memory access from "addr(size)" format
tracing::mem_access_t parse_mem_access(const std::string& str) {
  if (str.empty()) return {0, 0};

  size_t paren_pos = str.find('(');
  if (paren_pos == std::string::npos) {
    return {parse_hex(str), 0};
  }

  unsigned long addr = parse_hex(str.substr(0, paren_pos));
  unsigned int size =
      std::stoul(str.substr(paren_pos + 1, str.find(')') - paren_pos - 1));
  return {addr, size};
}

// Helper to split semicolon-separated list
std::vector<std::string> split_semicolon(const std::string& str) {
  std::vector<std::string> result;
  if (str.empty()) return result;

  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ';')) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

// Helper to parse semicolon-separated hex IPs
std::vector<unsigned long> parse_ip_list(const std::string& str) {
  std::vector<unsigned long> result;
  auto items = split_semicolon(str);
  for (const auto& item : items) {
    result.push_back(parse_hex(item));
  }
  return result;
}

// Helper to parse semicolon-separated memory accesses
std::vector<tracing::mem_access_t> parse_mem_access_list(
    const std::string& str) {
  std::vector<tracing::mem_access_t> result;
  auto items = split_semicolon(str);
  for (const auto& item : items) {
    result.push_back(parse_mem_access(item));
  }
  return result;
}

std::vector<tracing::instr_trace_t> parse_csv(const std::string& csv_path) {
  std::vector<tracing::instr_trace_t> instructions;

  try {
    io::CSVReader<16, io::trim_chars<' ', '\t'>,
                  io::double_quote_escape<',', '"'>>
        in(csv_path);

    in.read_header(
        io::ignore_extra_column, "IP", "Assembly", "Category", "Opcode",
        "Branch Type", "Branch Taken", "Branch Target Address",
        "Instruction Sync", "Read Registers", "Write Registers",
        "Register Dependent IPs", "Read Addresses", "Write Addresses",
        "Memory Dependent IPs", "Fetch Latency", "Execution Latency");

    std::string ip_str, assembly, category, opcode;
    std::string branch_type, branch_taken, branch_target_addr;
    std::string inst_sync_str;
    std::string read_regs_str, write_regs_str, reg_deps_str;
    std::string read_addrs_str, write_addrs_str, mem_deps_str;
    std::string fetch_latency_str, exec_latency_str;

    uint64_t row = 2;

    while (in.read_row(ip_str, assembly, category, opcode, branch_type,
                       branch_taken, branch_target_addr, inst_sync_str,
                       read_regs_str, write_regs_str, reg_deps_str,
                       read_addrs_str, write_addrs_str, mem_deps_str,
                       fetch_latency_str, exec_latency_str)) {
      tracing::instr_trace_t inst;

      inst.ip = parse_hex(ip_str);
      inst.assembly = assembly;
      inst.category = category;
      inst.opcode = opcode;
      inst.branch_type = branch_type;
      inst.branch_taken = (branch_taken == "true" || branch_taken == "True");
      inst.branch_target_addr = parse_hex(branch_target_addr);
      inst.inst_sync = (inst_sync_str == "true" || inst_sync_str == "True");

      inst.read_registers = split_semicolon(read_regs_str);
      inst.write_registers = split_semicolon(write_regs_str);
      inst.reg_dependent_ips = parse_ip_list(reg_deps_str);
      inst.read_addresses = parse_mem_access_list(read_addrs_str);
      inst.write_addresses = parse_mem_access_list(write_addrs_str);
      inst.mem_dependent_ips = parse_ip_list(mem_deps_str);

      if (fetch_latency_str.empty()) {
        throw std::runtime_error("Missing fetch latency at row: " +
                                 std::to_string(row));
      }

      inst.fetch_latency = std::stoul(fetch_latency_str);

      if (exec_latency_str.empty()) {
        throw std::runtime_error("Missing execution latency at row " +
                                 std::to_string(row));
      }

      inst.exe_latency = std::stoul(exec_latency_str);

      instructions.push_back(inst);
    }
  } catch (const io::error::base& e) {
    throw std::runtime_error(std::string("CSV parsing error: ") + e.what());
  }

  return instructions;
}

Instr convert_to_instr(const tracing::instr_trace_t& inst, instr_id_t id) {
  Instr result;

  result.IP = inst.ip;
  result.id = id;
  result.exe_latency = inst.exe_latency;
  result.fetch_latency = inst.fetch_latency;

  // Determine instruction type from opcode lookup table
  uint8_t opcat = get_opcode_categories(inst.opcode);
  result.is_alu = (opcat & OPCAT_ALU) != 0;
  result.is_mul = (opcat & OPCAT_MUL) != 0;
  result.is_div = (opcat & OPCAT_DIV) != 0;
  result.is_fp = (opcat & OPCAT_FP) != 0;
  result.is_fp_mult_div = (opcat & OPCAT_FP_MULT_DIV) != 0;

  result.is_load = !inst.read_addresses.empty();
  result.is_store = !inst.write_addresses.empty();

  // if the instruction was a load, store its read address
  if (result.is_load) {
    result.read_address = inst.read_addresses.front().addr;
  } else {
    result.read_address = 0;
  }

  result.is_isb = inst.inst_sync;

  // Determine branch type
  result.is_branch =
      (inst.category == "COND_BR" || inst.category == "UNCOND_BR" ||
       inst.category == "CALL" || inst.category == "RET");

  if (result.is_branch) {
    if (inst.branch_type.find("conditional") != std::string::npos) {
      result.branch_type = branch_t::DIRECT_COND;
    } else if (inst.branch_type.find("indirect") != std::string::npos) {
      result.branch_type = branch_t::INDIRECT;
    } else {
      result.branch_type = branch_t::DIRECT_UNCOND;
    }

    result.branch_taken = inst.branch_taken;
    result.branch_target_addr = inst.branch_target_addr;
  }

  return result;
}

std::vector<Instr> convert_trace(
    const std::vector<tracing::instr_trace_t>& instructions) {
  std::vector<Instr> result;
  result.reserve(instructions.size());

  // Map from IP to instruction ID for dependency resolution
  std::unordered_map<uint64_t, instr_id_t> ip_to_id;

  // First pass: convert instructions
  for (size_t i = 0; i < instructions.size(); ++i) {
    Instr instr = convert_to_instr(instructions[i], static_cast<instr_id_t>(i));
    ip_to_id[instr.IP] = instr.id;
    result.push_back(instr);
  }

  // Second pass: resolve dependencies
  for (size_t i = 0; i < instructions.size(); ++i) {
    const auto& trace_inst = instructions[i];
    auto& instr = result[i];

    // Add register dependencies
    for (uint64_t dep_ip : trace_inst.reg_dependent_ips) {
      auto it = ip_to_id.find(dep_ip);
      if (it != ip_to_id.end() && it->second < instr.id) {
        instr.deps.push_back(it->second);
      }
    }

    // Add memory dependencies
    for (uint64_t dep_ip : trace_inst.mem_dependent_ips) {
      auto it = ip_to_id.find(dep_ip);
      if (it != ip_to_id.end() && it->second < instr.id) {
        instr.deps.push_back(it->second);
      }
    }

    // Remove duplicates and sort
    std::sort(instr.deps.begin(), instr.deps.end());
    instr.deps.erase(std::unique(instr.deps.begin(), instr.deps.end()),
                     instr.deps.end());
  }

  return result;
}

std::vector<Instr> parse_and_convert(const std::string& csv_path) {
  auto trace = parse_csv(csv_path);
  return convert_trace(trace);
}

}  // namespace analytical

#ifdef STANDALONE_PARSER

#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <csv_file>" << std::endl;
    return 1;
  }

  try {
    std::cout << "Parsing CSV file: " << argv[1] << std::endl;

    // keep the parsed trace so we can print mem_latency alongside the converted
    // Instrs
    auto trace = analytical::parse_csv(argv[1]);
    auto instructions = analytical::convert_trace(trace);

    std::cout << "\nSuccessfully parsed " << instructions.size()
              << " instructions\n"
              << std::endl;

    // Print first 10 instructions as sample
    size_t print_count = std::min(size_t(10), instructions.size());
    std::cout << "First " << print_count << " instructions:\n" << std::endl;

    for (size_t i = 0; i < print_count; ++i) {
      const auto& instr = instructions[i];

      std::cout << "Instruction " << i << ":" << std::endl;
      std::cout << "  IP: 0x" << std::hex << instr.IP << std::dec << std::endl;
      std::cout << "  ID: " << instr.id << std::endl;
      // print both execution and memory latencies (memory latency comes from
      // the parsed trace)
      std::cout << "  Latencies: exe=" << instr.exe_latency
                << " mem=" << trace[i].mem_latency << std::endl;
      std::cout << "  Type: ";
      if (instr.is_alu) std::cout << "ALU ";
      if (instr.is_fp) std::cout << "FP ";
      if (instr.is_load) std::cout << "LOAD ";
      if (instr.is_store) std::cout << "STORE ";
      if (instr.is_branch) std::cout << "BRANCH ";
      if (instr.is_isb) std::cout << "ISB ";
      std::cout << std::endl;

      if (instr.is_branch) {
        std::cout << "  Branch Type: ";
        switch (instr.branch_type) {
          case analytical::branch_t::DIRECT_COND:
            std::cout << "Direct Conditional";
            break;
          case analytical::branch_t::DIRECT_UNCOND:
            std::cout << "Direct Unconditional";
            break;
          case analytical::branch_t::INDIRECT:
            std::cout << "Indirect";
            break;
        }
        std::cout << std::endl;
      }

      std::cout << "  Dependencies (" << instr.deps.size() << "): ";
      for (size_t j = 0; j < instr.deps.size(); ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << instr.deps[j];
      }
      std::cout << std::endl;
    }

    // Print statistics
    size_t alu_count = 0, fp_count = 0, load_count = 0, store_count = 0;
    size_t branch_count = 0, isb_count = 0;
    size_t total_deps = 0;

    for (const auto& instr : instructions) {
      if (instr.is_alu) alu_count++;
      if (instr.is_fp) fp_count++;
      if (instr.is_load) load_count++;
      if (instr.is_store) store_count++;
      if (instr.is_branch) branch_count++;
      if (instr.is_isb) isb_count++;
      total_deps += instr.deps.size();
    }

    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  ALU instructions: " << alu_count << std::endl;
    std::cout << "  FP instructions: " << fp_count << std::endl;
    std::cout << "  Load instructions: " << load_count << std::endl;
    std::cout << "  Store instructions: " << store_count << std::endl;
    std::cout << "  Branch instructions: " << branch_count << std::endl;
    std::cout << "  ISB instructions: " << isb_count << std::endl;
    std::cout << "  Average dependencies per instruction: "
              << (instructions.empty()
                      ? 0.0
                      : double(total_deps) / instructions.size())
              << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

#endif  // STANDALONE_PARSER