#include "pin.H"
extern "C"
{
#include "xed-interface.h"
}
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>

// This is more or less copied from the pintool examples

FILE* trace;
CONTEXT* ctx; 

typedef struct {
  unsigned long addr;
  UINT32 size;
} mem_access_t;

typedef struct {
  unsigned long addr;
  UINT32 size;
  unsigned long writer_ip;
} mem_write_t;

typedef struct {
  unsigned long ip;
  // std::string function;
  std::string assembly;
  std::string category;
  std::string opcode;
  std::string branch_type; // direct unconditional, direct conditional, indirect
  bool inst_sync;          // true if the instruction is a sync point
  std::vector<std::string> read_registers;
  std::vector<std::string> write_registers;
  std::vector<unsigned long> reg_dependent_ips;
  std::vector<unsigned long> mem_dependent_ips;
} instruction_data_t;

std::vector<mem_access_t> read_addresses;
std::vector<mem_access_t> write_addresses;
std::vector<mem_write_t> last_mem_writes;

// Global tracking variables
UINT32 num_inst_syncs = 0;
std::map<std::string, UINT32> branch_type_counts;


// Set of instructions that are sync points
std::set<unsigned long> sync_points = {
  // Fences / serializing
  XED_ICLASS_MFENCE,
  XED_ICLASS_SFENCE,
  XED_ICLASS_LFENCE,
  XED_ICLASS_CPUID,
  XED_ICLASS_SERIALIZE,
  XED_ICLASS_INVD,
  XED_ICLASS_WBINVD,
  // Return from interrupt (serializing)
  XED_ICLASS_IRET,
  XED_ICLASS_IRETD,
  XED_ICLASS_IRETQ,
  // Timestamp / MSR (serializing/ordered)
  XED_ICLASS_RDTSC,
  XED_ICLASS_RDTSCP,
  XED_ICLASS_RDMSR,
  XED_ICLASS_WRMSR,
  // Cache line flushes (often used with fences)
  XED_ICLASS_CLFLUSH,
  XED_ICLASS_CLFLUSHOPT,
  XED_ICLASS_CLWB,
  // Monitor/Wait/Spin-hint
  XED_ICLASS_MONITOR,
  XED_ICLASS_MONITORX,
  XED_ICLASS_MWAIT,
  XED_ICLASS_MWAITX,
  XED_ICLASS_PAUSE,
  // Transactional (TSX) boundaries
  XED_ICLASS_XBEGIN,
  XED_ICLASS_XEND,
  XED_ICLASS_XABORT,
  // Atomic read-modify-write
  XED_ICLASS_XCHG,
  XED_ICLASS_XADD,
  XED_ICLASS_XADD_LOCK,
  XED_ICLASS_CMPXCHG,
  XED_ICLASS_CMPXCHG_LOCK,
  XED_ICLASS_CMPXCHG8B,
  XED_ICLASS_CMPXCHG8B_LOCK,
  XED_ICLASS_CMPXCHG16B,
  XED_ICLASS_CMPXCHG16B_LOCK
};

// Map to track the last instruction ip that wrote to a register
std::map<std::string, unsigned long> last_reg_write_ip;



VOID log_read_op(VOID *ip, VOID *addr, UINT32 size) {
  mem_access_t access;
  access.addr = (unsigned long)addr;
  access.size = size;
  read_addresses.push_back(access);
}


VOID log_write_op(VOID *ip, VOID *addr, UINT32 size) {
  mem_access_t access;
  access.addr = (unsigned long)addr;
  access.size = size;
  write_addresses.push_back(access);
}

static inline bool ranges_overlap(unsigned long a, UINT32 as, unsigned long b, UINT32 bs) {
  unsigned long a_end = a + as;
  unsigned long b_end = b + bs;
  return !(a_end <= b || b_end <= a);
}

// Helper function to escape CSV fields (handle quotes and commas)
std::string escape_csv_field(const std::string& field) {
  std::string escaped;
  bool needs_quotes = false;
  
  for (char c : field) {
    if (c == '"') {
      escaped += "\"\"";
      needs_quotes = true;
    } else if (c == ',' || c == '\n' || c == '\r') {
      escaped += c;
      needs_quotes = true;
    } else {
      escaped += c;
    }
  }
  
  if (needs_quotes) {
    return "\"" + escaped + "\"";
  }
  return escaped;
}

VOID log_instruction(CONTEXT* ctx, instruction_data_t *id, xed_decoded_inst_t* xedd) {
  // Compute register dependencies at EXECUTION time (not instrumentation time)
  // This ensures dependencies are tracked based on execution order, not binary order
  for (const auto& reg : id->read_registers) {
    if (last_reg_write_ip.find(reg) != last_reg_write_ip.end()) {
      unsigned long dep_ip = last_reg_write_ip[reg];
      // Avoid self-dependencies (instruction reading and writing same register)
      if (dep_ip != id->ip) {
        if (std::find(id->reg_dependent_ips.begin(), id->reg_dependent_ips.end(), dep_ip) == id->reg_dependent_ips.end()) {
          id->reg_dependent_ips.push_back(dep_ip);
        }
      }
    }
  }
  
  // Update last write IP for written registers at EXECUTION time
  for (const auto& reg : id->write_registers) {
    last_reg_write_ip[reg] = id->ip;
  }
  
  // Compute memory dependencies by checking current reads against previous writes
  for (const auto& r : read_addresses) {
    for (const auto& w : last_mem_writes) {
      if (ranges_overlap(r.addr, r.size, w.addr, w.size)) {
        if (std::find(id->mem_dependent_ips.begin(), id->mem_dependent_ips.end(), w.writer_ip) == id->mem_dependent_ips.end()) {
          id->mem_dependent_ips.push_back(w.writer_ip);
        }
      }
    }
  }

  // Write CSV row: IP, Assembly, Category, Opcode, Branch Type, Instruction Sync,
  // Read Registers, Write Registers, Register Dependent IPs, Read Addresses, Write Addresses, Memory Dependent IPs
  
  // IP
  fprintf(trace, "0x%lx,", id->ip);
  
  // Assembly
  fprintf(trace, "%s,", escape_csv_field(id->assembly).c_str());
  
  // Category
  fprintf(trace, "%s,", escape_csv_field(id->category).c_str());
  
  // Opcode
  fprintf(trace, "%s,", escape_csv_field(id->opcode).c_str());
  
  // Branch Type
  fprintf(trace, "%s,", escape_csv_field(id->branch_type).c_str());
  
  // Instruction Sync
  fprintf(trace, "%s,", id->inst_sync ? "true" : "false");
  
  // Read Registers (semicolon-separated)
  if (id->read_registers.empty()) {
    fprintf(trace, ",");
  } else {
    for (size_t i = 0; i < id->read_registers.size(); i++) {
      if (i > 0) fprintf(trace, ";");
      fprintf(trace, "%s", id->read_registers[i].c_str());
    }
    fprintf(trace, ",");
  }
  
  // Write Registers (semicolon-separated)
  if (id->write_registers.empty()) {
    fprintf(trace, ",");
  } else {
    for (size_t i = 0; i < id->write_registers.size(); i++) {
      if (i > 0) fprintf(trace, ";");
      fprintf(trace, "%s", id->write_registers[i].c_str());
    }
    fprintf(trace, ",");
  }
  
  // Register Dependent IPs (semicolon-separated)
  if (id->reg_dependent_ips.empty()) {
    fprintf(trace, ",");
  } else {
    for (size_t i = 0; i < id->reg_dependent_ips.size(); i++) {
      if (i > 0) fprintf(trace, ";");
      fprintf(trace, "0x%lx", id->reg_dependent_ips[i]);
    }
    fprintf(trace, ",");
  }
  
  // Read Addresses (semicolon-separated, format: 0xADDR(SIZE))
  if (read_addresses.empty()) {
    fprintf(trace, ",");
  } else {
    for (size_t i = 0; i < read_addresses.size(); i++) {
      if (i > 0) fprintf(trace, ";");
      fprintf(trace, "0x%lx(%u)", read_addresses[i].addr, read_addresses[i].size);
    }
    fprintf(trace, ",");
  }
  
  // Write Addresses (semicolon-separated, format: 0xADDR(SIZE))
  if (write_addresses.empty()) {
    fprintf(trace, ",");
  } else {
    for (size_t i = 0; i < write_addresses.size(); i++) {
      if (i > 0) fprintf(trace, ";");
      fprintf(trace, "0x%lx(%u)", write_addresses[i].addr, write_addresses[i].size);
    }
    fprintf(trace, ",");
  }
  
  // Memory Dependent IPs (semicolon-separated)
  if (id->mem_dependent_ips.empty()) {
    fprintf(trace, "\n");
  } else {
    for (size_t i = 0; i < id->mem_dependent_ips.size(); i++) {
      if (i > 0) fprintf(trace, ";");
      fprintf(trace, "0x%lx", id->mem_dependent_ips[i]);
    }
    fprintf(trace, "\n");
  }

  // Update last written memory ranges for dependency of future instructions
  if (!write_addresses.empty()) {
    // Start from the existing set of disjoint (possibly adjacent) write ranges
    std::vector<mem_write_t> updated = last_mem_writes;

    for (const auto& curw : write_addresses) {
      std::vector<mem_write_t> next;
      next.reserve(updated.size() + 2);
      unsigned long c_start = curw.addr;
      unsigned long c_end   = curw.addr + curw.size;

      for (const auto& prev : updated) {
        unsigned long p_start = prev.addr;
        unsigned long p_end   = prev.addr + prev.size;

        if (!(c_end <= p_start || p_end <= c_start)) {
          // Overlap exists; split previous into up to two residual parts
          if (p_start < c_start) {
            // Left residual [p_start, c_start)
            next.push_back({p_start, (UINT32)(c_start - p_start), prev.writer_ip});
          }
          if (c_end < p_end) {
            // Right residual [c_end, p_end)
            next.push_back({c_end, (UINT32)(p_end - c_end), prev.writer_ip});
          }
          // Overlapped middle is fully covered by current write; dropped
        } else {
          // No overlap; keep previous interval
          next.push_back(prev);
        }
      }

      // Add the current write range as owned by this instruction
      next.push_back({c_start, (UINT32)(c_end - c_start), id->ip});

      // Move to next state for processing subsequent writes in the same instruction
      updated.swap(next);
    }

    last_mem_writes.swap(updated);
  }

  read_addresses.clear();
  write_addresses.clear();
}

VOID trace_instr(INS ins, VOID* v)
{

    instruction_data_t *id = new instruction_data_t();
    id->ip = INS_Address(ins);
    // id->function = RTN_FindNameByAddress(id->ip);
    id->assembly = INS_Disassemble(ins);
    id->category = CATEGORY_StringShort(INS_Category(ins));
    id->opcode = OPCODE_StringShort(INS_Opcode(ins));

    if (sync_points.find(INS_Opcode(ins)) != sync_points.end()) {
      id->inst_sync = true;
      num_inst_syncs++;
    }

    if (INS_IsBranch(ins)) {
      if (INS_IsDirectBranch(ins)) {
        if (INS_HasFallThrough(ins)) {
          id->branch_type = "direct conditional";
        } else {
          id->branch_type = "direct unconditional";
        }
      } else {
        id->branch_type = "indirect";
      }
    }
    branch_type_counts[id->branch_type]++;

    // Get registers read by the instruction (at instrumentation time)
    // Note: Register dependency tracking is done at EXECUTION time in log_instruction()
    UINT32 maxReadRegs = INS_MaxNumRRegs(ins);
    for (UINT32 i = 0; i < maxReadRegs; i++) {
        REG reg = INS_RegR(ins, i);
        std::string reg_name = REG_StringShort(reg);
        id->read_registers.push_back(reg_name);
        // Dependency tracking moved to log_instruction() for execution-time accuracy
    }

    // Get registers written by the instruction (at instrumentation time)
    // Note: last_reg_write_ip update is done at EXECUTION time in log_instruction()
    UINT32 maxWriteRegs = INS_MaxNumWRegs(ins);
    for (UINT32 i = 0; i < maxWriteRegs; i++) {
        REG reg = INS_RegW(ins, i);
        std::string reg_name = REG_StringShort(reg);
        id->write_registers.push_back(reg_name);
        // last_reg_write_ip update moved to log_instruction() for execution-time accuracy
    }

    // Potential calls to add:
    // INS_IsCacheLineFlush()
    // INS_OperandCount()
    // INS_OperandIsReg()
    // INS_OperandReg()
    // INS_OperandImmediate()
    // INS_OperandIsMemory()
    // INS_OperandMemoryBaseReg()
    // INS_OperandMemoryDisplacement()
    // INS_OperandMemoryIndexReg()
    // INS_OperandMemoryScale()
    // INS_OperandRead()
    // INS_OperandReadAndWritten()
    // INS_OperandReadOnly()
    // INS_OperandWritten()
    // INS_OperandWrittenOnly()

    UINT32 memOperands = INS_MemoryOperandCount(ins);
    for (UINT32 memOp = 0; memOp < memOperands; memOp++) {
      if (INS_MemoryOperandIsRead(ins, memOp)) {
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)log_read_op,
                               IARG_INST_PTR,
                               IARG_MEMORYOP_EA, memOp,
                               IARG_MEMORYOP_SIZE, memOp,
                               IARG_END);
      }
      if (INS_MemoryOperandIsWritten(ins, memOp)) {
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)log_write_op,
                               IARG_INST_PTR,
                               IARG_MEMORYOP_EA, memOp,
                               IARG_MEMORYOP_SIZE, memOp,
                               IARG_END);
      }
    }

    // Call the log_instruction function to log the instruction information
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)log_instruction, IARG_CONTEXT, IARG_PTR, id, IARG_END);

}

VOID Fini(INT32 code, VOID* v)
{	
	fclose(trace);
  
  trace = fopen("trace_summary.txt", "w");
  fprintf(trace, "Number of instruction sync points: %u\n", num_inst_syncs);
  fprintf(trace, "Number of direct unconditional branches: %u\n", branch_type_counts.count("direct unconditional") ? branch_type_counts["direct unconditional"] : 0);
  fprintf(trace, "Number of direct conditional branches: %u\n", branch_type_counts.count("direct conditional") ? branch_type_counts["direct conditional"] : 0);
  fprintf(trace, "Number of indirect branches: %u\n", branch_type_counts.count("indirect") ? branch_type_counts["indirect"] : 0);
  fclose(trace);
}


int main(int argc, char* argv[])
{
	PIN_InitSymbols();
	if (PIN_Init(argc, argv)) {
		std::cout << "ERROR: could not init pin..." << std::endl;
		return 1;
	}

	trace = fopen("trace.csv", "w");
	
	// Write CSV header
	fprintf(trace, "IP,Assembly,Category,Opcode,Branch Type,Instruction Sync,Read Registers,Write Registers,Register Dependent IPs,Read Addresses,Write Addresses,Memory Dependent IPs\n");

	INS_AddInstrumentFunction(trace_instr, 0);

	PIN_AddFiniFunction(Fini, 0);

	PIN_StartProgram();

	return 0;
}