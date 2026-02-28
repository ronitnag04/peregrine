#ifndef OPCODE_CATEGORIES_H
#define OPCODE_CATEGORIES_H

#include <cstdint>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Opcode category bitmask flags
//   ALU          - scalar integer arithmetic / logic / shift / bit-manipulation
//   ALU_MULT_DIV - scalar integer multiply and divide (disjoint from ALU)
//   SIMD         - integer SIMD ops on XMM/YMM registers (disjoint from FP)
//   FP           - scalar and packed floating-point ops (disjoint from SIMD)
//   FP_MULT_DIV  - FP multiply, divide, sqrt, FMA (slow FP ports, disjoint from FP)
//   LOAD         - reads from memory (inherently a load)
//   STORE        - writes to memory (inherently a store)
//   LOAD_STORE   - reads AND writes memory (atomic RMW / string move)
//
// Disjointness: an instruction belongs to exactly one of {ALU, ALU_MULT_DIV}
// and exactly one of {SIMD, FP, FP_MULT_DIV} (or none).
//
// Note: LOAD and STORE are set only for opcodes that are *inherently*
// load or store operations (non-temporal, string, gather/scatter, atomics).
// For opcodes whose memory direction depends on operand form (e.g. MOV,
// ADD), determine is_load / is_store from the trace's read/write address
// lists instead.
// ---------------------------------------------------------------------------
enum OpcodeCategory : uint8_t {
  OPCAT_NONE = 0x00,
  OPCAT_ALU = 0x01,
  OPCAT_ALU_MULT_DIV = 0x02,  // scalar integer multiply and divide
  OPCAT_SIMD = 0x04,          // integer SIMD (XMM/YMM, disjoint from FP)
  OPCAT_FP = 0x08,            // scalar/packed float (disjoint from SIMD)
  OPCAT_LOAD = 0x10,
  OPCAT_STORE = 0x20,
  OPCAT_LOAD_STORE = 0x30,   // OPCAT_LOAD | OPCAT_STORE
  OPCAT_FP_MULT_DIV = 0x40,  // FP multiply, divide, sqrt, FMA (slow FP units)
};

// ---------------------------------------------------------------------------
// Lookup table: opcode string (uppercase) -> category bitmask
// ---------------------------------------------------------------------------
inline const std::unordered_map<std::string, uint8_t>& opcode_category_map() {
  static const std::unordered_map<std::string, uint8_t> TABLE = {

      // ===================================================================
      //  ALU  –  integer arithmetic, logic, shifts, bit manipulation
      // ===================================================================

      // Data transfer (register forms use integer execution ports)
      {"MOV", OPCAT_ALU},
      {"MOVSX", OPCAT_ALU},
      {"MOVZX", OPCAT_ALU},
      {"MOVSXD", OPCAT_ALU},
      {"CMOVB", OPCAT_ALU},
      {"CMOVNB", OPCAT_ALU},
      {"CMOVZ", OPCAT_ALU},
      {"CMOVNZ", OPCAT_ALU},
      {"CMOVS", OPCAT_ALU},
      {"CMOVNS", OPCAT_ALU},
      {"CMOVBE", OPCAT_ALU},
      {"CMOVNBE", OPCAT_ALU},
      {"BSWAP", OPCAT_ALU},
      {"MOVBE", OPCAT_ALU},
      {"LEA", OPCAT_ALU},
      {"PUSH", OPCAT_ALU},
      {"POP", OPCAT_ALU},
      {"LEAVE", OPCAT_ALU},

      // Arithmetic
      {"ADD", OPCAT_ALU},
      {"SUB", OPCAT_ALU},
      {"ADC", OPCAT_ALU},
      {"SBB", OPCAT_ALU},
      {"INC", OPCAT_ALU},
      {"DEC", OPCAT_ALU},
      {"NEG", OPCAT_ALU},
      {"CMP", OPCAT_ALU},
      {"ADCX", OPCAT_ALU},
      {"ADOX", OPCAT_ALU},

      // Logic
      {"AND", OPCAT_ALU},
      {"OR", OPCAT_ALU},
      {"XOR", OPCAT_ALU},
      {"NOT", OPCAT_ALU},
      {"TEST", OPCAT_ALU},

      // Shifts and rotates
      {"SHR", OPCAT_ALU},
      {"SHL", OPCAT_ALU},
      {"SAR", OPCAT_ALU},
      {"ROR", OPCAT_ALU},
      {"ROL", OPCAT_ALU},
      {"RCR", OPCAT_ALU},
      {"RCL", OPCAT_ALU},
      {"SHRD", OPCAT_ALU},
      {"SHLD", OPCAT_ALU},
      {"SHLX", OPCAT_ALU},
      {"SHRX", OPCAT_ALU},
      {"SARX", OPCAT_ALU},

      // Bit manipulation
      {"BT", OPCAT_ALU},
      {"BTR", OPCAT_ALU},
      {"BTS", OPCAT_ALU},
      {"BTC", OPCAT_ALU},
      {"BSF", OPCAT_ALU},
      {"BSR", OPCAT_ALU},
      {"LZCNT", OPCAT_ALU},
      {"TZCNT", OPCAT_ALU},
      {"POPCNT", OPCAT_ALU},
      {"BLSI", OPCAT_ALU},
      {"BLSMSK", OPCAT_ALU},
      {"BLSR", OPCAT_ALU},
      {"BEXTR", OPCAT_ALU},
      {"BZHI", OPCAT_ALU},
      {"PDEP", OPCAT_ALU},
      {"PEXT", OPCAT_ALU},
      {"ANDN", OPCAT_ALU},

      // Setcc / flags
      {"SETZ", OPCAT_ALU},
      {"SETNZ", OPCAT_ALU},
      {"SETNB", OPCAT_ALU},
      {"SETO", OPCAT_ALU},
      {"SETNLE", OPCAT_ALU},
      {"CLC", OPCAT_ALU},
      {"STC", OPCAT_ALU},
      {"CMC", OPCAT_ALU},
      {"LAHF", OPCAT_ALU},
      {"SAHF", OPCAT_ALU},

      // Scalar integer conversions
      {"CBW", OPCAT_ALU},
      {"CWDE", OPCAT_ALU},
      {"CDQE", OPCAT_ALU},
      {"CWD", OPCAT_ALU},
      {"CDQ", OPCAT_ALU},
      {"CQO", OPCAT_ALU},

      // Misc scalar / system (execute on integer ports)
      {"NOP", OPCAT_ALU},
      {"SYSCALL", OPCAT_ALU},
      {"CPUID", OPCAT_ALU},
      {"RDTSC", OPCAT_ALU},
      {"RDTSCP", OPCAT_ALU},
      {"RDRAND", OPCAT_ALU},
      {"RDSEED", OPCAT_ALU},
      {"XGETBV", OPCAT_ALU},

      // Memory fences / hints (no data movement, integer ports)
      {"LFENCE", OPCAT_ALU},
      {"MFENCE", OPCAT_ALU},
      {"SFENCE", OPCAT_ALU},
      {"PAUSE", OPCAT_ALU},

      // Control transfer (branch condition evaluated on ALU port)
      {"JMP", OPCAT_ALU},
      {"CALL", OPCAT_ALU},
      {"CALL_NEAR", OPCAT_ALU},
      {"RET", OPCAT_ALU},
      {"RET_NEAR", OPCAT_ALU},
      {"LOOP", OPCAT_ALU},
      {"JZ", OPCAT_ALU},
      {"JE", OPCAT_ALU},
      {"JNZ", OPCAT_ALU},
      {"JNE", OPCAT_ALU},
      {"JG", OPCAT_ALU},
      {"JGE", OPCAT_ALU},
      {"JL", OPCAT_ALU},
      {"JNL", OPCAT_ALU},
      {"JLE", OPCAT_ALU},
      {"JNLE", OPCAT_ALU},
      {"JA", OPCAT_ALU},
      {"JAE", OPCAT_ALU},
      {"JS", OPCAT_ALU},
      {"JNS", OPCAT_ALU},
      {"JB", OPCAT_ALU},
      {"JP", OPCAT_ALU},
      {"JNB", OPCAT_ALU},
      {"JBE", OPCAT_ALU},
      {"JNBE", OPCAT_ALU},
      {"JECXZ", OPCAT_ALU},
      {"JRCXZ", OPCAT_ALU},

      // ===================================================================
      //  ALU_MULT_DIV  –  integer multiply and divide (disjoint from ALU)
      // ===================================================================
      {"MUL", OPCAT_ALU_MULT_DIV},
      {"IMUL", OPCAT_ALU_MULT_DIV},
      {"MULX", OPCAT_ALU_MULT_DIV},
      {"DIV", OPCAT_ALU_MULT_DIV},
      {"IDIV", OPCAT_ALU_MULT_DIV},

      // ===================================================================
      //  SIMD  –  integer SIMD operations on XMM/YMM/ZMM registers.
      //           Disjoint from FP: data is treated as integers, not floats.
      // ===================================================================

      // Integer vector move
      {"MOVD", OPCAT_SIMD},
      {"MOVQ", OPCAT_SIMD},
      {"MOVDQA", OPCAT_SIMD},
      {"MOVDQU", OPCAT_SIMD},
      {"VMOVD", OPCAT_SIMD},
      {"VMOVDQA", OPCAT_SIMD},
      {"VMOVDQU", OPCAT_SIMD},
      {"LDDQU", OPCAT_SIMD | OPCAT_LOAD},
      {"MOVNTQ", OPCAT_SIMD | OPCAT_STORE},
      {"MOVNTDQ", OPCAT_SIMD | OPCAT_STORE},
      {"MOVNTDQA", OPCAT_SIMD | OPCAT_LOAD},
      {"PMOVMSKB", OPCAT_SIMD},
      {"VPMOVMSKB", OPCAT_SIMD},

      // Pack / unpack
      {"PACKSSWB", OPCAT_SIMD},
      {"PACKSSDW", OPCAT_SIMD},
      {"PACKUSWB", OPCAT_SIMD},
      {"PACKUSDW", OPCAT_SIMD},
      {"PUNPCKHBW", OPCAT_SIMD},
      {"PUNPCKLBW", OPCAT_SIMD},
      {"PUNPCKLWD", OPCAT_SIMD},
      {"PUNPCKLDQ", OPCAT_SIMD},
      {"PUNPCKLQDQ", OPCAT_SIMD},

      // Shuffle / permute
      {"PSHUFB", OPCAT_SIMD},
      {"PSHUFD", OPCAT_SIMD},
      {"PSHUFW", OPCAT_SIMD},
      {"PSHUFLW", OPCAT_SIMD},
      {"PSHUFHW", OPCAT_SIMD},
      {"PALIGNR", OPCAT_SIMD},
      {"VALIGND", OPCAT_SIMD},
      {"VALIGNQ", OPCAT_SIMD},
      {"VPERMD", OPCAT_SIMD},
      {"VPERMQ", OPCAT_SIMD},
      {"VPERM2I128", OPCAT_SIMD},

      // Insert / extract
      {"PEXTRB", OPCAT_SIMD},
      {"PEXTRW", OPCAT_SIMD},
      {"PEXTRD", OPCAT_SIMD},
      {"PEXTRQ", OPCAT_SIMD},
      {"PINSRB", OPCAT_SIMD},
      {"PINSRW", OPCAT_SIMD},
      {"PINSRD", OPCAT_SIMD},
      {"PINSRQ", OPCAT_SIMD},
      {"VINSERTI128", OPCAT_SIMD},
      {"VEXTRACTI128", OPCAT_SIMD},

      // Broadcast / gather / scatter (integer)
      {"VPBROADCASTB", OPCAT_SIMD},
      {"VPBROADCASTD", OPCAT_SIMD},
      {"VPBROADCASTQ", OPCAT_SIMD},
      {"VPGATHERDD", OPCAT_SIMD | OPCAT_LOAD},
      {"VPGATHERDQ", OPCAT_SIMD | OPCAT_LOAD},
      {"VPSCATTERDD", OPCAT_SIMD | OPCAT_STORE},
      {"VPSCATTERDQ", OPCAT_SIMD | OPCAT_STORE},

      // Integer vector arithmetic
      {"PADDB", OPCAT_SIMD},
      {"PADDW", OPCAT_SIMD},
      {"PADDD", OPCAT_SIMD},
      {"PADDQ", OPCAT_SIMD},
      {"PSUBB", OPCAT_SIMD},
      {"PSUBW", OPCAT_SIMD},
      {"PSUBD", OPCAT_SIMD},
      {"PSUBQ", OPCAT_SIMD},
      {"PADDSB", OPCAT_SIMD},
      {"PADDSW", OPCAT_SIMD},
      {"PADDUSB", OPCAT_SIMD},
      {"PADDUSW", OPCAT_SIMD},
      {"PHADDW", OPCAT_SIMD},
      {"PHADDD", OPCAT_SIMD},
      {"PHSUBW", OPCAT_SIMD},
      {"PHSUBD", OPCAT_SIMD},
      {"PAVGB", OPCAT_SIMD},
      {"PAVGW", OPCAT_SIMD},
      {"PABSB", OPCAT_SIMD},
      {"PABSW", OPCAT_SIMD},
      {"PABSD", OPCAT_SIMD},
      {"PMINUB", OPCAT_SIMD},
      {"VPMINUB", OPCAT_SIMD},
      {"PMAXUB", OPCAT_SIMD},

      // Integer vector multiply
      {"PMULLW", OPCAT_SIMD},
      {"PMULLD", OPCAT_SIMD},
      {"PMULHW", OPCAT_SIMD},
      {"PMULHUW", OPCAT_SIMD},
      {"PMULUDQ", OPCAT_SIMD},
      {"PMULDQ", OPCAT_SIMD},
      {"PMADDWD", OPCAT_SIMD},
      {"PMADDUBSW", OPCAT_SIMD},
      {"VPDPBUSD", OPCAT_SIMD},

      // Integer vector compare
      {"PCMPEQB", OPCAT_SIMD},
      {"PCMPEQW", OPCAT_SIMD},
      {"PCMPEQD", OPCAT_SIMD},
      {"PCMPEQQ", OPCAT_SIMD},
      {"PCMPGTB", OPCAT_SIMD},
      {"PCMPGTW", OPCAT_SIMD},
      {"PCMPGTD", OPCAT_SIMD},
      {"PCMPGTQ", OPCAT_SIMD},
      {"VPCMPEQB", OPCAT_SIMD},

      // Integer vector logical
      {"PAND", OPCAT_SIMD},
      {"PANDN", OPCAT_SIMD},
      {"POR", OPCAT_SIMD},
      {"PXOR", OPCAT_SIMD},
      {"VPOR", OPCAT_SIMD},
      {"VPXOR", OPCAT_SIMD},
      {"PTEST", OPCAT_SIMD},
      {"VPTEST", OPCAT_SIMD},

      // Integer vector shift
      {"PSLLW", OPCAT_SIMD},
      {"PSLLD", OPCAT_SIMD},
      {"PSLLQ", OPCAT_SIMD},
      {"PSRLW", OPCAT_SIMD},
      {"PSRLD", OPCAT_SIMD},
      {"PSRLQ", OPCAT_SIMD},
      {"PSRAW", OPCAT_SIMD},
      {"PSRAD", OPCAT_SIMD},
      {"PSLLDQ", OPCAT_SIMD},
      {"PSRLDQ", OPCAT_SIMD},

      // Crypto / string SIMD
      {"PCMPESTRI", OPCAT_SIMD},
      {"PCMPESTRM", OPCAT_SIMD},
      {"PCMPISTRI", OPCAT_SIMD},
      {"PCMPISTRM", OPCAT_SIMD},
      {"AESENC", OPCAT_SIMD},
      {"AESENCLAST", OPCAT_SIMD},
      {"AESDEC", OPCAT_SIMD},
      {"AESDECLAST", OPCAT_SIMD},
      {"PCLMULQDQ", OPCAT_SIMD},
      {"SHA1RNDS4", OPCAT_SIMD},
      {"SHA256RNDS2", OPCAT_SIMD},

      // ===================================================================
      //  FP  –  scalar and packed floating-point operations.
      //         Disjoint from SIMD: data is treated as floats.
      // ===================================================================

      // FP scalar / packed move
      {"MOVAPS", OPCAT_FP},
      {"MOVAPD", OPCAT_FP},
      {"MOVUPS", OPCAT_FP},
      {"MOVSS", OPCAT_FP},
      {"MOVSD", OPCAT_FP},
      {"MOVSD_XMM", OPCAT_FP},
      {"MOVHPS", OPCAT_FP},
      {"MOVHPD", OPCAT_FP},
      {"MOVLPS", OPCAT_FP},
      {"MOVLPD", OPCAT_FP},
      {"MOVHLPS", OPCAT_FP},
      {"MOVLHPS", OPCAT_FP},
      {"MOVNTPS", OPCAT_FP | OPCAT_STORE},
      {"MOVNTPD", OPCAT_FP | OPCAT_STORE},
      {"MOVDDUP", OPCAT_FP},
      {"MOVSHDUP", OPCAT_FP},
      {"MOVSLDUP", OPCAT_FP},

      // FP shuffle / permute / blend
      {"SHUFPS", OPCAT_FP},
      {"SHUFPD", OPCAT_FP},
      {"VPERMILPS", OPCAT_FP},
      {"VPERMILPD", OPCAT_FP},
      {"VPERM2F128", OPCAT_FP},
      {"BLENDPS", OPCAT_FP},
      {"BLENDPD", OPCAT_FP},
      {"BLENDVPS", OPCAT_FP},
      {"BLENDVPD", OPCAT_FP},

      // FP insert / extract
      {"EXTRACTPS", OPCAT_FP},
      {"INSERTPS", OPCAT_FP},
      {"VINSERTF128", OPCAT_FP},
      {"VEXTRACTF128", OPCAT_FP},

      // FP broadcast / gather / scatter
      {"VBROADCASTSS", OPCAT_FP},
      {"VBROADCASTSD", OPCAT_FP},
      {"VGATHERDPS", OPCAT_FP | OPCAT_LOAD},
      {"VGATHERQPS", OPCAT_FP | OPCAT_LOAD},
      {"VSCATTERDPS", OPCAT_FP | OPCAT_STORE},

      // FP arithmetic
      {"ADDSS", OPCAT_FP},
      {"ADDSD", OPCAT_FP},
      {"ADDPS", OPCAT_FP},
      {"ADDPD", OPCAT_FP},
      {"SUBSS", OPCAT_FP},
      {"SUBSD", OPCAT_FP},
      {"SUBPS", OPCAT_FP},
      {"SUBPD", OPCAT_FP},
      {"MULSS", OPCAT_FP_MULT_DIV},
      {"MULSD", OPCAT_FP_MULT_DIV},
      {"MULPS", OPCAT_FP_MULT_DIV},
      {"MULPD", OPCAT_FP_MULT_DIV},
      {"DIVSS", OPCAT_FP_MULT_DIV},
      {"DIVSD", OPCAT_FP_MULT_DIV},
      {"DIVPS", OPCAT_FP_MULT_DIV},
      {"DIVPD", OPCAT_FP_MULT_DIV},
      {"SQRTSS", OPCAT_FP_MULT_DIV},
      {"SQRTSD", OPCAT_FP_MULT_DIV},
      {"SQRTPS", OPCAT_FP_MULT_DIV},
      {"SQRTPD", OPCAT_FP_MULT_DIV},
      {"RSQRTSS", OPCAT_FP_MULT_DIV},
      {"RSQRTPS", OPCAT_FP_MULT_DIV},
      {"RCPSS", OPCAT_FP_MULT_DIV},
      {"RCPPS", OPCAT_FP_MULT_DIV},

      // FMA
      {"VFMADD132PS", OPCAT_FP_MULT_DIV},
      {"VFMADD213PS", OPCAT_FP_MULT_DIV},
      {"VFMADD231PS", OPCAT_FP_MULT_DIV},

      // FP compare / logic
      {"CMPSS", OPCAT_FP},
      {"CMPSD", OPCAT_FP},
      {"CMPPS", OPCAT_FP},
      {"CMPPD", OPCAT_FP},
      {"COMISS", OPCAT_FP},
      {"COMISD", OPCAT_FP},
      {"UCOMISS", OPCAT_FP},
      {"UCOMISD", OPCAT_FP},
      {"MAXSS", OPCAT_FP},
      {"MAXSD", OPCAT_FP},
      {"MAXPS", OPCAT_FP},
      {"MAXPD", OPCAT_FP},
      {"MINSS", OPCAT_FP},
      {"MINSD", OPCAT_FP},
      {"MINPS", OPCAT_FP},
      {"MINPD", OPCAT_FP},
      {"ANDPS", OPCAT_FP},
      {"ANDPD", OPCAT_FP},
      {"ORPS", OPCAT_FP},
      {"ORPD", OPCAT_FP},
      {"XORPS", OPCAT_FP},
      {"XORPD", OPCAT_FP},

      // FP conversions
      {"CVTPD2PS", OPCAT_FP},
      {"CVTPS2PD", OPCAT_FP},
      {"CVTSS2SD", OPCAT_FP},
      {"CVTSD2SS", OPCAT_FP},
      {"CVTPS2DQ", OPCAT_FP},
      {"CVTDQ2PS", OPCAT_FP},
      {"CVTTPS2DQ", OPCAT_FP},
      {"CVTSS2SI", OPCAT_FP},
      {"CVTSD2SI", OPCAT_FP},
      {"CVTSI2SS", OPCAT_FP},
      {"CVTSI2SD", OPCAT_FP},

      // FP math / special
      {"ROUNDPS", OPCAT_FP},
      {"ROUNDPD", OPCAT_FP},
      {"DPPS", OPCAT_FP},
      {"DPPD", OPCAT_FP},
      {"VZEROUPPER", OPCAT_FP},
      {"FNSTCW", OPCAT_FP},

      // ===================================================================
      //  LOAD  –  inherently memory-read (no separate store form)
      // ===================================================================
      {"LODS", OPCAT_LOAD},
      {"XLAT", OPCAT_LOAD | OPCAT_ALU},
      {"XRSTOR", OPCAT_LOAD},  // restores extended CPU state from memory

      // ===================================================================
      //  STORE  –  inherently memory-write (no separate load form)
      // ===================================================================
      {"STOS", OPCAT_STORE},
      {"STOSQ", OPCAT_STORE},
      {"MOVNTI", OPCAT_STORE},
      {"XSAVE", OPCAT_STORE},  // saves extended CPU state to memory
      {"XSAVEC", OPCAT_STORE},

      // ===================================================================
      //  LOAD | STORE  –  atomic RMW or string ops that read and write
      // ===================================================================
      {"XCHG", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"XADD", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"CMPXCHG", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"LOCK_XADD", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"LOCK_ADD", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"CMPXCHG_LOCK", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"MOVS", OPCAT_LOAD_STORE},
      {"SCAS", OPCAT_LOAD_STORE | OPCAT_ALU},
      {"CMPS", OPCAT_LOAD_STORE | OPCAT_ALU},
  };
  return TABLE;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

inline uint8_t get_opcode_categories(const std::string& opcode) {
  const auto& m = opcode_category_map();
  auto it = m.find(opcode);
  return (it != m.end()) ? it->second : OPCAT_NONE;
}

inline bool opcode_is_alu(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_ALU;
}
inline bool opcode_is_alu_mult_div(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_ALU_MULT_DIV;
}
inline bool opcode_is_simd(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_SIMD;
}
inline bool opcode_is_fp(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_FP;
}
inline bool opcode_is_fp_mult_div(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_FP_MULT_DIV;
}
inline bool opcode_is_load(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_LOAD;
}
inline bool opcode_is_store(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_STORE;
}
// "load/store" — uses the LS issue port (either a load or a store or both)
inline bool opcode_is_ls(const std::string& op) {
  return get_opcode_categories(op) & (OPCAT_LOAD | OPCAT_STORE);
}

#endif  // OPCODE_CATEGORIES_H
