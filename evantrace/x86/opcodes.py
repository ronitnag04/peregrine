from enum import Enum


class Opcode(Enum):
    """
    Maps x86 opcodes to gem5 OpClass sequences per microop.

    Each opcode value is a dict keyed by variant ('reg_reg', 'mem_reg', 'reg_mem').
    Each variant maps to a list of OpClass strings: one entry per microop in execution
    order (from FuncUnitConfig).
    Derived from gem5 x86 ISA: insts/general_purpose/*.py macroop definitions.
    Variants: reg_reg ≈ _R_R, mem_reg ≈ _M_R/_P_R (dest mem), reg_mem ≈ _R_M/_R_P (src mem).
    Microop→OpClass: ld→MemRead, st→MemWrite, ldst→MemRead, add/sub/and/or/xor/mov/limm/rdip→IntAlu,
    mul1s/mul1u→IntMult, mulel/muleh→IntAlu, div1/div2→IntDiv, mfence/lfence/sfence→System.
    """

    # =========================================================================
    #                           SCALAR INSTRUCTIONS
    # =========================================================================

    # --- Data Transfer ---
    MOV     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemWrite'], 'reg_mem': ['MemRead'] }
    MOVSX   = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    MOVZX   = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    MOVSXD  = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    CMOVB   = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVNB  = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVZ   = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVNZ  = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVS   = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVNS  = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVBE  = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    CMOVNBE = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'] }
    XCHG    = { 'reg_reg': ['IntAlu', 'IntAlu', 'IntAlu'], 'mem_reg': ['System', 'MemRead', 'MemWrite', 'System', 'IntAlu'], 'reg_mem': ['System', 'MemRead', 'MemWrite', 'System', 'IntAlu'] }
    BSWAP   = { 'reg_reg': ['IntAlu'] }
    MOVBE   = { 'mem_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['IntAlu', 'MemWrite'] }
    MOVNTI  = { 'reg_mem': ['MemWrite'] }

    # --- Stack & Address ---
    PUSH    = { 'reg_reg': ['MemWrite', 'IntAlu'], 'mem_reg': ['MemRead', 'MemWrite', 'IntAlu'] }
    POP     = { 'reg_reg': ['MemRead', 'IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'IntAlu', 'MemWrite'] }
    LEA     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['IntAlu'] }
    XLAT    = { 'reg_reg': ['IntAlu', 'MemRead'] }

    # --- Arithmetic (Basic) ---
    ADD     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    SUB     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'MemWrite'], 'reg_mem': ['MemRead', 'IntAlu'] }
    ADC     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'MemWrite'], 'reg_mem': ['MemRead', 'IntAlu'] }
    SBB     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'MemWrite'], 'reg_mem': ['MemRead', 'IntAlu'] }
    INC     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    DEC     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    NEG     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    CMP     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    ADCX    = { 'reg_reg': ['IntAlu'] }
    ADOX    = { 'reg_reg': ['IntAlu'] }

    # --- Arithmetic (Multiplication/Division) ---
    MUL     = { 'reg_reg': ['IntMult', 'IntAlu', 'IntAlu'], 'reg_mem': ['MemRead', 'IntMult', 'IntAlu', 'IntAlu'] }
    IMUL    = { 'reg_reg': ['IntMult', 'IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntMult', 'IntAlu', 'IntAlu'], 'reg_mem': ['MemRead', 'IntMult', 'IntAlu', 'IntAlu'] }
    MULX    = { 'reg_reg': ['IntMult'] }
    DIV     = { 'reg_reg': ['IntDiv'], 'mem_reg': ['MemRead', 'IntDiv'] }
    IDIV    = { 'reg_reg': ['IntDiv'], 'mem_reg': ['MemRead', 'IntDiv'] }

    # --- Logic & Bitwise ---
    AND     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'MemWrite'], 'reg_mem': ['MemRead', 'IntAlu'] }
    OR      = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'MemWrite'], 'reg_mem': ['MemRead', 'IntAlu'] }
    XOR     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'MemWrite'], 'reg_mem': ['MemRead', 'IntAlu'] }
    NOT     = { 'reg_reg': ['IntAlu', 'IntAlu'], 'reg_mem': ['IntAlu', 'MemRead', 'IntAlu', 'MemWrite'] }
    TEST    = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }

    # --- Shifts & Rotates ---
    SHR     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    SHL     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    SAR     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    ROR     = { 'reg_reg': ['IntAlu'] }
    ROL     = { 'reg_reg': ['IntAlu'] }
    RCR     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    RCL     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    SHRD    = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    SHLD    = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    SHLX    = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    SHRX    = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    SARX    = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }

    # --- Bit Manipulation ---
    BT      = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    BTR     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    BTS     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    BTC     = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    BSF     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    BSR     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    LZCNT   = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    TZCNT   = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    POPCNT  = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    BLSI    = { 'reg_reg': ['IntAlu'] }
    BLSMSK  = { 'reg_reg': ['IntAlu'] }
    BLSR    = { 'reg_reg': ['IntAlu'] }
    BEXTR   = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    BZHI    = { 'reg_reg': ['IntAlu'] }
    PDEP    = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    PEXT    = { 'reg_reg': ['IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    ANDN    = { 'reg_reg': ['IntAlu'] }

    # --- Flags & System ---
    SETZ    = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['IntAlu', 'IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'IntAlu', 'MemWrite'] }
    SETNZ   = { 'reg_reg': ['IntAlu', 'IntAlu'], 'reg_mem': ['IntAlu', 'IntAlu', 'MemWrite'] }
    SETNB   = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['IntAlu', 'IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'IntAlu', 'MemWrite'] }
    SETNP   = { 'reg_reg': ['IntAlu', 'IntAlu'], 'mem_reg': ['IntAlu', 'IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'IntAlu', 'MemWrite'] }
    SETO    = { 'reg_reg': ['IntAlu', 'IntAlu'], 'reg_mem': ['IntAlu', 'IntAlu', 'MemWrite'] }
    SETNLE  = { 'reg_reg': ['IntAlu', 'IntAlu'], 'reg_mem': ['IntAlu', 'IntAlu', 'MemWrite'] }
    CLC     = { 'reg_reg': ['IntAlu'] }
    STC     = { 'reg_reg': ['IntAlu'] }
    CMC     = { 'reg_reg': ['IntAlu'] }
    LAHF    = { 'reg_reg': ['IntAlu'] }
    SAHF    = { 'reg_reg': ['IntAlu'] }
    SYSCALL = { 'reg_reg': ['System'] }

    # --- Control Transfer ---
    JMP       = { 'reg_reg': ['IntAlu'], 'mem_reg': ['MemRead', 'IntAlu'] }
    CALL      = { 'reg_reg': ['IntAlu', 'MemWrite', 'IntAlu', 'IntAlu'], 'mem_reg': ['IntAlu', 'MemRead', 'MemWrite', 'IntAlu', 'IntAlu'] } 
    CALL_NEAR = { 'reg_reg': ['IntAlu', 'MemWrite', 'IntAlu', 'IntAlu'], 'mem_reg': ['IntAlu', 'MemRead', 'MemWrite', 'IntAlu', 'IntAlu'] } 
    RET       = { 'reg_reg': ['MemRead', 'IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'IntAlu'] }
    RET_NEAR  = { 'reg_reg': ['MemRead', 'IntAlu', 'IntAlu'], 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu', 'IntAlu'] }
    LOOP      = { 'reg_reg': ['IntAlu', 'IntAlu'] }
    JZ        = { 'reg_reg': ['IntAlu'] }
    JE        = { 'reg_reg': ['IntAlu'] }
    JNZ       = { 'reg_reg': ['IntAlu'] }
    JNE       = { 'reg_reg': ['IntAlu'] }
    JG        = { 'reg_reg': ['IntAlu'] }
    JGE       = { 'reg_reg': ['IntAlu'] }
    JL        = { 'reg_reg': ['IntAlu'] }
    JNL       = { 'reg_reg': ['IntAlu'] }
    JLE       = { 'reg_reg': ['IntAlu'] }
    JNLE      = { 'reg_reg': ['IntAlu'] }
    JA        = { 'reg_reg': ['IntAlu'] }
    JAE       = { 'reg_reg': ['IntAlu'] }
    JS        = { 'reg_reg': ['IntAlu'] }
    JNS       = { 'reg_reg': ['IntAlu'] }
    JB        = { 'reg_reg': ['IntAlu'] }
    JP        = { 'reg_reg': ['IntAlu'] }
    JNB       = { 'reg_reg': ['IntAlu'] }
    JBE       = { 'reg_reg': ['IntAlu'] }
    JNBE      = { 'reg_reg': ['IntAlu'] }
    JECXZ     = { 'reg_reg': ['IntAlu'], 'mem_reg': ['IntAlu'], 'reg_mem': ['IntAlu'] }
    JRCXZ     = { 'reg_reg': ['IntAlu'] }

    # --- Strings (Scalar) ---
    LODS    = { 'reg_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    STOS    = { 'reg_reg': ['IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'MemWrite'] }
    STOSB   = { 'reg_reg': ['IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'MemWrite'] }
    STOSQ   = { 'reg_reg': ['IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'MemWrite'] }
    MOVS    = { 'reg_reg': ['MemRead', 'MemWrite'] }
    SCAS    = { 'reg_reg': ['MemRead', 'IntAlu'], 'reg_mem': ['MemRead', 'IntAlu'] }
    CMPS    = { 'reg_reg': ['MemRead', 'MemRead', 'IntAlu'] }

    # --- Atomic/Sync ---
    XADD          = { 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite', 'IntAlu'] }
    CMPXCHG       = { 'reg_mem': ['MemRead', 'IntAlu', 'IntAlu', 'MemWrite', 'IntAlu'] }
    LOCK_XADD     = { 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite', 'IntAlu'] }
    LOCK_ADD      = { 'reg_mem': ['MemRead', 'IntAlu', 'MemWrite'] }
    CMPXCHG_LOCK  = { 'mem_reg': ['MemRead', 'IntAlu', 'IntAlu', 'MemWrite', 'IntAlu'] }
    LFENCE        = { 'reg_reg': ['System'] }
    MFENCE        = { 'reg_reg': ['System'] }
    SFENCE        = { 'reg_reg': ['System'] }
    PAUSE         = { 'reg_reg': ['System'] }
    LEAVE         = { 'mem_reg': ['IntAlu', 'MemRead', 'IntAlu', 'IntAlu'] }

    # --- Conversion (Scalar) ---
    CBW     = { 'reg_reg': ['IntAlu'] }
    CWDE    = { 'reg_reg': ['IntAlu'] }
    CDQE    = { 'reg_reg': ['IntAlu'] }
    CWD     = { 'reg_reg': ['IntAlu'] }
    CDQ     = { 'reg_reg': ['IntAlu'] }
    CQO     = { 'reg_reg': ['IntAlu'] }

    # --- Misc ---
    NOP     = { 'reg_reg': ['IntAlu'] }
    CPUID   = { 'reg_reg': ['System'] }
    RDTSC   = { 'reg_reg': ['System'] }
    RDTSCP  = { 'reg_reg': ['System'] }
    RDRAND  = { 'reg_reg': ['System'] }
    RDSEED  = { 'reg_reg': ['System'] }
    XGETBV  = { 'reg_reg': ['System'] }

    # =========================================================================
    #                        INTEGER VECTOR INSTRUCTIONS
    # =========================================================================

    # --- Vector Move ---
    MOVD      = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    MOVQ      = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['MemRead', 'SimdMisc'], 'reg_mem': ['SimdMisc', 'MemWrite'] }
    MOVDQA    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    MOVDQU    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    VMOVD     = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    VMOVDQA   = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    VMOVDQU   = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    VMOVQ     = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad'], 'reg_mem': ['SimdUnitStrideStore'] }
    LDDQU     = { 'mem_reg': ['SimdUnitStrideLoad'] }
    MOVNTQ    = { 'reg_mem': ['SimdUnitStrideStore'] }
    MOVNTDQ   = { 'reg_mem': ['SimdUnitStrideStore'] }
    MOVNTDQA  = { 'mem_reg': ['SimdUnitStrideLoad'] }
    PMOVMSKB  = { 'reg_reg': ['SimdMisc'] }
    VPMOVMSKB = { 'reg_reg': ['SimdMisc'] }

    # --- Pack / Unpack ---
    PACKSSWB   = { 'reg_reg': ['SimdAlu'] }
    PACKSSDW   = { 'reg_reg': ['SimdAlu'] }
    PACKUSWB   = { 'reg_reg': ['SimdAlu'] }
    PACKUSDW   = { 'reg_reg': ['SimdAlu'] }
    PUNPCKHBW  = { 'reg_reg': ['SimdMisc'] }
    PUNPCKLBW  = { 'reg_reg': ['SimdMisc'] }
    PUNPCKLWD  = { 'reg_reg': ['SimdMisc'] }
    PUNPCKLDQ  = { 'reg_reg': ['SimdMisc'] }
    PUNPCKLQDQ = { 'reg_reg': ['SimdMisc'] }

    # --- Shuffle / Permute ---
    PSHUFB    = { 'reg_reg': ['SimdMisc'] }
    PSHUFD    = { 'reg_reg': ['SimdMisc'] }
    PSHUFW    = { 'reg_reg': ['SimdMisc'] }
    PSHUFLW   = { 'reg_reg': ['SimdMisc'] }
    PSHUFHW   = { 'reg_reg': ['SimdMisc'] }
    PALIGNR   = { 'reg_reg': ['SimdMisc'] }
    VALIGND   = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VALIGNQ   = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VPERMD    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VPERMQ    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VPERM2I128= { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }

    # --- Insert / Extract ---
    PEXTRB    = { 'reg_reg': ['SimdMisc'], 'reg_mem': ['SimdMisc', 'MemWrite'] }
    PEXTRW    = { 'reg_reg': ['SimdMisc'], 'reg_mem': ['SimdMisc', 'MemWrite'] }
    PEXTRD    = { 'reg_reg': ['SimdMisc'], 'reg_mem': ['SimdMisc', 'MemWrite'] }
    PEXTRQ    = { 'reg_reg': ['SimdMisc'], 'reg_mem': ['SimdMisc', 'MemWrite'] }
    PINSRB    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['MemRead', 'SimdMisc'] }
    PINSRW    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['MemRead', 'SimdMisc'] }
    PINSRD    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['MemRead', 'SimdMisc'] }
    PINSRQ    = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['MemRead', 'SimdMisc'] }
    VINSERTI128 = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VEXTRACTI128 = { 'reg_reg': ['SimdMisc'], 'reg_mem': ['SimdMisc', 'SimdUnitStrideStore'] }

    # --- Broadcast / Gather / Scatter ---
    VPBROADCASTB = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VPBROADCASTD = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VPBROADCASTQ = { 'reg_reg': ['SimdMisc'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdMisc'] }
    VPGATHERDD   = { 'mem_reg': ['SimdIndexedLoad', 'SimdMisc'] }
    VPGATHERDQ   = { 'mem_reg': ['SimdIndexedLoad', 'SimdMisc'] }
    VPSCATTERDD  = { 'reg_mem': ['SimdMisc', 'SimdIndexedStore'] }
    VPSCATTERDQ  = { 'reg_mem': ['SimdMisc', 'SimdIndexedStore'] }

    # --- Vector Arithmetic ---
    PADDB     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDW     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDD     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDQ     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PSUBB     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PSUBW     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PSUBD     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PSUBQ     = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDSB    = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDSW    = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDUSB   = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PADDUSW   = { 'reg_reg': ['SimdAdd'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAdd'] }
    PHADDW    = { 'reg_reg': ['SimdAdd'] }
    PHADDD    = { 'reg_reg': ['SimdAdd'] }
    PHSUBW    = { 'reg_reg': ['SimdAdd'] }
    PHSUBD    = { 'reg_reg': ['SimdAdd'] }
    PAVGB     = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdFloatAlu'] }
    PAVGW     = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdFloatAlu'] }
    PABSB     = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    PABSW     = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    PABSD     = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    PMINUB    = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    VPMINUB   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PMAXUB    = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }

    # --- Vector Multiply ---
    PMULLW    = { 'reg_reg': ['SimdMult'] }
    PMULLD    = { 'reg_reg': ['SimdMult'] }
    PMULHW    = { 'reg_reg': ['SimdMult'] }
    PMULHUW   = { 'reg_reg': ['SimdMult'] }
    PMULUDQ   = { 'reg_reg': ['SimdMult'] }
    PMULDQ    = { 'reg_reg': ['SimdMult'] }
    PMADDWD   = { 'reg_reg': ['SimdMultAcc'] }
    PMADDUBSW = { 'reg_reg': ['SimdMultAcc'] }
    VPDPBUSD  = { 'reg_reg': ['SimdMultAcc'] }

    # --- Vector Compare ---
    PCMPEQB   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPEQW   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPEQD   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPEQQ   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPGTB   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPGTW   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPGTD   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    PCMPGTQ   = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }
    VPCMPEQB  = { 'reg_reg': ['SimdCmp'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdCmp'] }

    # --- Vector Logical ---
    PAND      = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    PANDN     = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    POR       = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    PXOR      = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    VPOR      = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    VPXOR     = { 'reg_reg': ['SimdAlu'], 'mem_reg': ['SimdUnitStrideLoad', 'SimdAlu'] }
    PTEST     = { 'reg_reg': ['SimdCmp'] }
    VPTEST    = { 'reg_reg': ['SimdCmp'] }

    # --- Vector Shift ---
    PSLLW     = { 'reg_reg': ['SimdShift'] }
    PSLLD     = { 'reg_reg': ['SimdShift'] }
    PSLLQ     = { 'reg_reg': ['SimdShift'] }
    PSRLW     = { 'reg_reg': ['SimdShift'] }
    PSRLD     = { 'reg_reg': ['SimdShift'] }
    PSRLQ     = { 'reg_reg': ['SimdShift'] }
    PSRAW     = { 'reg_reg': ['SimdShift'] }
    PSRAD     = { 'reg_reg': ['SimdShift'] }
    PSLLDQ    = { 'reg_reg': ['SimdShift'] }
    PSRLDQ    = { 'reg_reg': ['SimdShift'] }

    # --- Encryption / String Support ---
    PCMPESTRI = { 'reg_reg': ['SimdCmp', 'SimdMisc'] }
    PCMPESTRM = { 'reg_reg': ['SimdCmp', 'SimdMisc'] }
    PCMPISTRI = { 'reg_reg': ['SimdCmp', 'SimdMisc'] }
    PCMPISTRM = { 'reg_reg': ['SimdCmp', 'SimdMisc'] }
    AESENC    = { 'reg_reg': ['SimdAes'] }
    AESENCLAST= { 'reg_reg': ['SimdAes'] }
    AESDEC    = { 'reg_reg': ['SimdAes'] }
    AESDECLAST= { 'reg_reg': ['SimdAes'] }
    PCLMULQDQ = { 'reg_reg': ['SimdAesMix'] }
    SHA1RNDS4 = { 'reg_reg': ['SimdSha1Hash'] }
    SHA256RNDS2 = { 'reg_reg': ['SimdSha256Hash'] }

    # =========================================================================
    #                   FLOATING POINT VECTOR INSTRUCTIONS
    # =========================================================================

    # --- Data Movement ---
    MOVAPS    = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVAPD    = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVUPS    = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVSS     = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVSD     = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVSD_XMM = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    VMOVSD    = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVHPS    = { 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVHPD    = { 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVLPS    = { 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVLPD    = { 'mem_reg': ['FloatMemRead'], 'reg_mem': ['FloatMemWrite'] }
    MOVHLPS   = { 'reg_reg': ['SimdFloatMisc'] }
    MOVLHPS   = { 'reg_reg': ['SimdFloatMisc'] }
    MOVNTPS   = { 'reg_mem': ['FloatMemWrite'] }
    MOVNTPD   = { 'reg_mem': ['FloatMemWrite'] }
    MOVDDUP   = { 'reg_reg': ['SimdFloatCvt'], 'mem_reg': ['FloatMemRead', 'SimdFloatCvt'] }
    MOVSHDUP  = { 'reg_reg': ['SimdFloatMisc'] }
    MOVSLDUP  = { 'reg_reg': ['SimdFloatMisc'] }

    # --- Shuffle / Permute / Blend ---
    SHUFPS    = { 'reg_reg': ['SimdFloatMisc'] }
    SHUFPD    = { 'reg_reg': ['SimdFloatMisc'] }
    VPERMILPS = { 'reg_reg': ['SimdFloatMisc'] }
    VPERMILPD = { 'reg_reg': ['SimdFloatMisc'] }
    VPERM2F128= { 'reg_reg': ['SimdFloatMisc'] }
    BLENDPS   = { 'reg_reg': ['SimdFloatAlu'] }
    BLENDPD   = { 'reg_reg': ['SimdFloatAlu'] }
    BLENDVPS  = { 'reg_reg': ['SimdFloatAlu'] }
    BLENDVPD  = { 'reg_reg': ['SimdFloatAlu'] }

    # --- Insert / Extract ---
    EXTRACTPS = { 'reg_reg': ['SimdFloatMisc'], 'reg_mem': ['SimdFloatMisc', 'MemWrite'] }
    INSERTPS  = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMisc'] }
    VINSERTF128  = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMisc'] }
    VEXTRACTF128 = { 'reg_reg': ['SimdFloatMisc'], 'reg_mem': ['SimdFloatMisc', 'FloatMemWrite'] }

    # --- Broadcast / Gather / Scatter (Float) ---
    VBROADCASTSS = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMisc'] }
    VBROADCASTSD = { 'reg_reg': ['SimdFloatMisc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMisc'] }
    VGATHERDPS   = { 'mem_reg': ['SimdIndexedLoad', 'SimdFloatMisc'] }
    VGATHERQPS   = { 'mem_reg': ['SimdIndexedLoad', 'SimdFloatMisc'] }
    VSCATTERDPS  = { 'reg_mem': ['SimdFloatMisc', 'SimdIndexedStore'] }

    # --- Arithmetic (Float) ---
    ADDSS     = { 'reg_reg': ['FloatAdd'], 'mem_reg': ['FloatMemRead', 'FloatAdd'] }
    ADDSD     = { 'reg_reg': ['FloatAdd'], 'mem_reg': ['FloatMemRead', 'FloatAdd'] }
    VADDSD    = { 'reg_reg': ['FloatAdd'], 'mem_reg': ['FloatMemRead', 'FloatAdd'] }
    ADDPS     = { 'reg_reg': ['SimdFloatAdd'], 'mem_reg': ['FloatMemRead', 'SimdFloatAdd'] }
    ADDPD     = { 'reg_reg': ['SimdFloatAdd'], 'mem_reg': ['FloatMemRead', 'SimdFloatAdd'] }
    SUBSS     = { 'reg_reg': ['FloatAdd'], 'mem_reg': ['FloatMemRead', 'FloatAdd'] }
    SUBSD     = { 'reg_reg': ['FloatAdd'], 'mem_reg': ['FloatMemRead', 'FloatAdd'] }
    VSUBSD    = { 'reg_reg': ['FloatAdd'], 'mem_reg': ['FloatMemRead', 'FloatAdd'] }
    SUBPS     = { 'reg_reg': ['SimdFloatAdd'], 'mem_reg': ['FloatMemRead', 'SimdFloatAdd'] }
    SUBPD     = { 'reg_reg': ['SimdFloatAdd'], 'mem_reg': ['FloatMemRead', 'SimdFloatAdd'] }
    MULSS     = { 'reg_reg': ['FloatMult'], 'mem_reg': ['FloatMemRead', 'FloatMult'] }
    MULSD     = { 'reg_reg': ['FloatMult'], 'mem_reg': ['FloatMemRead', 'FloatMult'] }
    VMULSD    = { 'reg_reg': ['FloatMult'], 'mem_reg': ['FloatMemRead', 'FloatMult'] }
    MULPS     = { 'reg_reg': ['SimdFloatMult'], 'mem_reg': ['FloatMemRead', 'SimdFloatMult'] }
    MULPD     = { 'reg_reg': ['SimdFloatMult'], 'mem_reg': ['FloatMemRead', 'SimdFloatMult'] }
    DIVSS     = { 'reg_reg': ['FloatDiv'] }
    DIVSD     = { 'reg_reg': ['FloatDiv'], 'mem_reg': ['FloatMemRead', 'FloatDiv'] } 
    DIVPS     = { 'reg_reg': ['SimdFloatDiv'] }
    DIVPD     = { 'reg_reg': ['SimdFloatDiv'] }
    SQRTSS    = { 'reg_reg': ['FloatSqrt'] }
    SQRTSD    = { 'reg_reg': ['FloatSqrt'] }
    SQRTPS    = { 'reg_reg': ['SimdFloatSqrt'] }
    SQRTPD    = { 'reg_reg': ['SimdFloatSqrt'] }
    RSQRTSS   = { 'reg_reg': ['FloatMisc'] }
    RSQRTPS   = { 'reg_reg': ['SimdFloatMisc'] }
    RCPSS     = { 'reg_reg': ['FloatMisc'] }
    RCPPS     = { 'reg_reg': ['SimdFloatMisc'] }

    # --- FMA (Fused Multiply-Add) ---
    VFMADD132PS = { 'reg_reg': ['SimdFloatMultAcc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMultAcc'] }
    VFMADD213PS = { 'reg_reg': ['SimdFloatMultAcc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMultAcc'] }
    VFMADD231PS = { 'reg_reg': ['SimdFloatMultAcc'], 'mem_reg': ['FloatMemRead', 'SimdFloatMultAcc'] }
    VFMADD132SD = { 'reg_reg': ['FloatMultAcc'], 'mem_reg': ['FloatMemRead', 'FloatMultAcc'] }
    VFMADD213SD = { 'reg_reg': ['FloatMultAcc'], 'mem_reg': ['FloatMemRead', 'FloatMultAcc'] }
    VFMADD231SD = { 'reg_reg': ['FloatMultAcc'], 'mem_reg': ['FloatMemRead', 'FloatMultAcc'] }
    VFNMADD132SD = { 'reg_reg': ['FloatMultAcc'], 'mem_reg': ['FloatMemRead', 'FloatMultAcc'] }

    # --- Comparison / Logic (Float) ---
    CMPSS     = { 'reg_reg': ['FloatCmp'] } 
    CMPSD     = { 'reg_reg': ['FloatCmp'] } 
    CMPPS     = { 'reg_reg': ['SimdFloatCmp'] } 
    CMPPD     = { 'reg_reg': ['SimdFloatCmp'] }
    COMISS    = { 'reg_reg': ['FloatCmp'] } 
    COMISD    = { 'reg_reg': ['FloatCmp'] } 
    UCOMISS   = { 'reg_reg': ['FloatCmp'] } 
    UCOMISD   = { 'reg_reg': ['FloatCmp'] } 
    MAXSS     = { 'reg_reg': ['FloatCmp'], 'mem_reg': ['FloatMemRead', 'FloatCmp'] }
    MAXSD     = { 'reg_reg': ['FloatCmp'], 'mem_reg': ['FloatMemRead', 'FloatCmp'] }
    MAXPS     = { 'reg_reg': ['SimdFloatCmp'], 'mem_reg': ['FloatMemRead', 'SimdFloatCmp'] }
    MAXPD     = { 'reg_reg': ['SimdFloatCmp'], 'mem_reg': ['FloatMemRead', 'SimdFloatCmp'] }
    MINSS     = { 'reg_reg': ['FloatCmp'], 'mem_reg': ['FloatMemRead', 'FloatCmp'] }
    MINSD     = { 'reg_reg': ['FloatCmp'], 'mem_reg': ['FloatMemRead', 'FloatCmp'] }
    MINPS     = { 'reg_reg': ['SimdFloatCmp'], 'mem_reg': ['FloatMemRead', 'SimdFloatCmp'] }
    MINPD     = { 'reg_reg': ['SimdFloatCmp'], 'mem_reg': ['FloatMemRead', 'SimdFloatCmp'] }
    ANDPS     = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] } 
    ANDPD     = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] } 
    ORPS      = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] }
    ORPD      = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] }
    XORPS     = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] }
    XORPD     = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] }
    VXORPS    = { 'reg_reg': ['SimdFloatAlu'], 'mem_reg': ['FloatMemRead', 'SimdFloatAlu'] }

    # --- Conversions (Float <-> Int/Float) ---
    CVTPD2PS  = { 'reg_reg': ['SimdFloatCvt'] }
    CVTPS2PD  = { 'reg_reg': ['SimdFloatCvt'] }
    CVTSS2SD  = { 'reg_reg': ['FloatCvt'] }
    CVTSD2SS  = { 'reg_reg': ['FloatCvt'] }
    CVTPS2DQ  = { 'reg_reg': ['SimdFloatCvt'] }
    CVTDQ2PS  = { 'reg_reg': ['SimdFloatCvt'] }
    CVTTPS2DQ = { 'reg_reg': ['SimdFloatCvt'] }
    CVTSS2SI  = { 'reg_reg': ['FloatCvt'] }
    CVTSD2SI  = { 'reg_reg': ['FloatCvt'] }
    CVTSI2SS  = { 'reg_reg': ['FloatCvt'] }
    CVTSI2SD  = { 'reg_reg': ['FloatCvt'] }
    VCVTSI2SD = { 'reg_reg': ['FloatCvt'] }

    # --- Math / Special ---
    ROUNDPS   = { 'reg_reg': ['SimdFloatMisc'] }
    ROUNDPD   = { 'reg_reg': ['SimdFloatMisc'] }
    DPPS      = { 'reg_reg': ['SimdFloatMultAcc'] }
    DPPD      = { 'reg_reg': ['SimdFloatMultAcc'] }

    # --- Other ---
    VZEROUPPER  = { 'reg_reg': ['SimdFloatMisc'] }
    XRSTOR      = { 'mem_reg': ['MemRead', 'System'] }
    XSAVE       = { 'mem_reg': ['MemWrite', 'System'] }
    XSAVEC      = { 'reg_mem': ['MemRead', 'MemWrite', 'System'] }

    FNSTCW      = { 'reg_reg': ['IntAlu'], 'mem_reg': ['IntAlu', 'MemWrite'], 'reg_mem': ['IntAlu', 'MemWrite'] }

    def latency(self, variant):
        custom_latencies = {
            "IntMult": 3,
            "IntDiv": 20,
            "FloatAdd": 2,
            "FloatCmp": 2,
            "FloatCvt": 2,
            "Bf16Cvt": 2,
            "FloatMult": 4,
            "FloatMultAcc": 5,
            "FloatMisc": 3,
            "FloatDiv": 12,
            "FloatSqrt": 24,
        }

        op_latency = 0
        opClasses = self.value[variant]
        for opClass in opClasses:
            op_latency += custom_latencies.get(opClass, 1)
        return op_latency


