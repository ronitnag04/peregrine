#ifndef OPCODE_CATEGORIES_H
#define OPCODE_CATEGORIES_H

#include <cstdint>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Opcode category bitmask flags
//   ALU       - integer arithmetic / logic / shift / bit-manipulation
//   MUL       - integer multiply
//   DIV       - integer divide
//   FP        - floating-point and SIMD / vector
//   LOAD      - reads from memory (inherently a load)
//   STORE     - writes to memory (inherently a store)
//   LOAD_STORE - reads AND writes memory (atomic RMW / string move)
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
  OPCAT_MUL = 0x02,
  OPCAT_DIV = 0x04,
  OPCAT_FP = 0x08,
  OPCAT_LOAD = 0x10,
  OPCAT_STORE = 0x20,
  OPCAT_LOAD_STORE = 0x30,  // OPCAT_LOAD | OPCAT_STORE
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
      //  MUL  –  integer multiply
      // ===================================================================
      {"MUL", OPCAT_MUL},
      {"IMUL", OPCAT_MUL},
      {"MULX", OPCAT_MUL},

      // ===================================================================
      //  DIV  –  integer divide
      // ===================================================================
      {"DIV", OPCAT_DIV},
      {"IDIV", OPCAT_DIV},

      // ===================================================================
      //  FP  –  floating-point scalar, SSE/AVX vector (integer and float)
      //         These all issue on the FP/vector execution ports.
      // ===================================================================

      // --- Integer vector move ---
      {"MOVD", OPCAT_FP},
      {"MOVQ", OPCAT_FP},
      {"MOVDQA", OPCAT_FP},
      {"MOVDQU", OPCAT_FP},
      {"VMOVD", OPCAT_FP},
      {"VMOVDQA", OPCAT_FP},
      {"VMOVDQU", OPCAT_FP},
      {"LDDQU", OPCAT_FP | OPCAT_LOAD},
      {"MOVNTQ", OPCAT_FP | OPCAT_STORE},
      {"MOVNTDQ", OPCAT_FP | OPCAT_STORE},
      {"MOVNTDQA", OPCAT_FP | OPCAT_LOAD},
      {"PMOVMSKB", OPCAT_FP},
      {"VPMOVMSKB", OPCAT_FP},

      // Pack / unpack
      {"PACKSSWB", OPCAT_FP},
      {"PACKSSDW", OPCAT_FP},
      {"PACKUSWB", OPCAT_FP},
      {"PACKUSDW", OPCAT_FP},
      {"PUNPCKHBW", OPCAT_FP},
      {"PUNPCKLBW", OPCAT_FP},
      {"PUNPCKLWD", OPCAT_FP},
      {"PUNPCKLDQ", OPCAT_FP},
      {"PUNPCKLQDQ", OPCAT_FP},

      // Shuffle / permute
      {"PSHUFB", OPCAT_FP},
      {"PSHUFD", OPCAT_FP},
      {"PSHUFW", OPCAT_FP},
      {"PSHUFLW", OPCAT_FP},
      {"PSHUFHW", OPCAT_FP},
      {"PALIGNR", OPCAT_FP},
      {"VALIGND", OPCAT_FP},
      {"VALIGNQ", OPCAT_FP},
      {"VPERMD", OPCAT_FP},
      {"VPERMQ", OPCAT_FP},
      {"VPERM2I128", OPCAT_FP},

      // Insert / extract
      {"PEXTRB", OPCAT_FP},
      {"PEXTRW", OPCAT_FP},
      {"PEXTRD", OPCAT_FP},
      {"PEXTRQ", OPCAT_FP},
      {"PINSRB", OPCAT_FP},
      {"PINSRW", OPCAT_FP},
      {"PINSRD", OPCAT_FP},
      {"PINSRQ", OPCAT_FP},
      {"VINSERTI128", OPCAT_FP},
      {"VEXTRACTI128", OPCAT_FP},

      // Broadcast / gather / scatter
      {"VPBROADCASTB", OPCAT_FP},
      {"VPBROADCASTD", OPCAT_FP},
      {"VPBROADCASTQ", OPCAT_FP},
      {"VPGATHERDD", OPCAT_FP | OPCAT_LOAD},
      {"VPGATHERDQ", OPCAT_FP | OPCAT_LOAD},
      {"VPSCATTERDD", OPCAT_FP | OPCAT_STORE},
      {"VPSCATTERDQ", OPCAT_FP | OPCAT_STORE},

      // Integer vector arithmetic
      {"PADDB", OPCAT_FP},
      {"PADDW", OPCAT_FP},
      {"PADDD", OPCAT_FP},
      {"PADDQ", OPCAT_FP},
      {"PSUBB", OPCAT_FP},
      {"PSUBW", OPCAT_FP},
      {"PSUBD", OPCAT_FP},
      {"PSUBQ", OPCAT_FP},
      {"PADDSB", OPCAT_FP},
      {"PADDSW", OPCAT_FP},
      {"PADDUSB", OPCAT_FP},
      {"PADDUSW", OPCAT_FP},
      {"PHADDW", OPCAT_FP},
      {"PHADDD", OPCAT_FP},
      {"PHSUBW", OPCAT_FP},
      {"PHSUBD", OPCAT_FP},
      {"PAVGB", OPCAT_FP},
      {"PAVGW", OPCAT_FP},
      {"PABSB", OPCAT_FP},
      {"PABSW", OPCAT_FP},
      {"PABSD", OPCAT_FP},
      {"PMINUB", OPCAT_FP},
      {"VPMINUB", OPCAT_FP},
      {"PMAXUB", OPCAT_FP},

      // Integer vector multiply
      {"PMULLW", OPCAT_FP},
      {"PMULLD", OPCAT_FP},
      {"PMULHW", OPCAT_FP},
      {"PMULHUW", OPCAT_FP},
      {"PMULUDQ", OPCAT_FP},
      {"PMULDQ", OPCAT_FP},
      {"PMADDWD", OPCAT_FP},
      {"PMADDUBSW", OPCAT_FP},
      {"VPDPBUSD", OPCAT_FP},

      // Integer vector compare
      {"PCMPEQB", OPCAT_FP},
      {"PCMPEQW", OPCAT_FP},
      {"PCMPEQD", OPCAT_FP},
      {"PCMPEQQ", OPCAT_FP},
      {"PCMPGTB", OPCAT_FP},
      {"PCMPGTW", OPCAT_FP},
      {"PCMPGTD", OPCAT_FP},
      {"PCMPGTQ", OPCAT_FP},
      {"VPCMPEQB", OPCAT_FP},

      // Integer vector logical
      {"PAND", OPCAT_FP},
      {"PANDN", OPCAT_FP},
      {"POR", OPCAT_FP},
      {"PXOR", OPCAT_FP},
      {"VPOR", OPCAT_FP},
      {"VPXOR", OPCAT_FP},
      {"PTEST", OPCAT_FP},
      {"VPTEST", OPCAT_FP},

      // Integer vector shift
      {"PSLLW", OPCAT_FP},
      {"PSLLD", OPCAT_FP},
      {"PSLLQ", OPCAT_FP},
      {"PSRLW", OPCAT_FP},
      {"PSRLD", OPCAT_FP},
      {"PSRLQ", OPCAT_FP},
      {"PSRAW", OPCAT_FP},
      {"PSRAD", OPCAT_FP},
      {"PSLLDQ", OPCAT_FP},
      {"PSRLDQ", OPCAT_FP},

      // Crypto / string support
      {"PCMPESTRI", OPCAT_FP},
      {"PCMPESTRM", OPCAT_FP},
      {"PCMPISTRI", OPCAT_FP},
      {"PCMPISTRM", OPCAT_FP},
      {"AESENC", OPCAT_FP},
      {"AESENCLAST", OPCAT_FP},
      {"AESDEC", OPCAT_FP},
      {"AESDECLAST", OPCAT_FP},
      {"PCLMULQDQ", OPCAT_FP},
      {"SHA1RNDS4", OPCAT_FP},
      {"SHA256RNDS2", OPCAT_FP},

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
      {"MULSS", OPCAT_FP},
      {"MULSD", OPCAT_FP},
      {"MULPS", OPCAT_FP},
      {"MULPD", OPCAT_FP},
      {"DIVSS", OPCAT_FP},
      {"DIVSD", OPCAT_FP},
      {"DIVPS", OPCAT_FP},
      {"DIVPD", OPCAT_FP},
      {"SQRTSS", OPCAT_FP},
      {"SQRTSD", OPCAT_FP},
      {"SQRTPS", OPCAT_FP},
      {"SQRTPD", OPCAT_FP},
      {"RSQRTSS", OPCAT_FP},
      {"RSQRTPS", OPCAT_FP},
      {"RCPSS", OPCAT_FP},
      {"RCPPS", OPCAT_FP},

      // FMA
      {"VFMADD132PS", OPCAT_FP},
      {"VFMADD213PS", OPCAT_FP},
      {"VFMADD231PS", OPCAT_FP},

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
inline bool opcode_is_mul(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_MUL;
}
inline bool opcode_is_div(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_DIV;
}
inline bool opcode_is_fp(const std::string& op) {
  return get_opcode_categories(op) & OPCAT_FP;
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
