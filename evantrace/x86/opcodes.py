from enum import Enum

class Opcode(Enum):
    """
    Maps x86 opcodes to estimated latencies (in cycles) for Ice Lake Architecture.
    Data derived from Agner Fog's Instruction Tables (Ice Lake).
    """
    
    # =========================================================================
    #                           SCALAR INSTRUCTIONS
    # =========================================================================

    # --- Data Transfer ---
    MOV     = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 2 } 
    MOVSX   = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 2 } 
    MOVZX   = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 2 } 
    MOVSXD  = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 2 } 
    CMOVB   = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVNB  = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVZ   = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVNZ  = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVS   = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVNS  = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVBE  = { 'reg_reg': 1, 'mem_reg': 4 }
    CMOVNBE = { 'reg_reg': 1, 'mem_reg': 4 }
    XCHG    = { 'reg_reg': 2, 'mem_reg': 19, 'reg_mem': 19 } 
    BSWAP   = { 'reg_reg': 1 } 
    MOVBE   = { 'mem_reg': 2, 'reg_mem': 3 } 
    MOVNTI  = { 'reg_mem': 400 } 

    # --- Stack & Address ---
    PUSH    = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 3 } 
    POP     = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 3 } 
    LEA     = { 'reg_reg': 1, 'mem_reg': 1 } 
    XLAT    = { 'reg_reg': 7 } 

    # --- Arithmetic (Basic) ---
    ADD     = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    SUB     = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    ADC     = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    SBB     = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    INC     = { 'reg_reg': 1, 'reg_mem': 7 } 
    DEC     = { 'reg_reg': 1, 'reg_mem': 7 } 
    NEG     = { 'reg_reg': 1, 'reg_mem': 7 } 
    CMP     = { 'reg_reg': 1, 'reg_mem': 7, 'mem_reg': 5 } 
    ADCX    = { 'reg_reg': 1 } 
    ADOX    = { 'reg_reg': 1 } 

    # --- Arithmetic (Multiplication/Division) ---
    MUL     = { 'reg_reg': 3, 'reg_mem': 7 } 
    IMUL    = { 'reg_reg': 3, 'mem_reg': 4, 'reg_mem': 7 }
    MULX    = { 'reg_reg': 4 } 
    DIV     = { 'reg_reg': 15, 'mem_reg': 19 } 
    IDIV    = { 'reg_reg': 15, 'mem_reg': 19 } 

    # --- Logic & Bitwise ---
    AND     = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    OR      = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    XOR     = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    NOT     = { 'reg_reg': 1, 'reg_mem': 7 }
    TEST    = { 'reg_reg': 1, 'mem_reg': 5, 'reg_mem': 7 }
    
    # --- Shifts & Rotates ---
    SHR     = { 'reg_reg': 1, 'reg_mem': 6 }
    SHL     = { 'reg_reg': 1, 'reg_mem': 6 }
    SAR     = { 'reg_reg': 1, 'reg_mem': 6 }
    ROR     = { 'reg_reg': 1 }
    ROL     = { 'reg_reg': 1 }
    RCR     = { 'reg_reg': 2, 'mem_reg': 6 }
    RCL     = { 'reg_reg': 2, 'mem_reg': 7 }
    SHRD    = { 'reg_reg': 3, 'reg_mem': 3 }
    SHLD    = { 'reg_reg': 3, 'reg_mem': 3 }
    SHLX    = { 'reg_reg': 1, 'mem_reg': 5 }
    SHRX    = { 'reg_reg': 1, 'mem_reg': 5 }
    SARX    = { 'reg_reg': 1, 'mem_reg': 5 }

    # --- Bit Manipulation ---
    BT      = { 'reg_reg': 1, 'reg_mem': 5 } 
    BTR     = { 'reg_reg': 1, 'reg_mem': 5 }
    BTS     = { 'reg_reg': 1, 'reg_mem': 5 }
    BTC     = { 'reg_reg': 1, 'reg_mem': 5 }
    BSF     = { 'reg_reg': 3, 'mem_reg': 7 }
    BSR     = { 'reg_reg': 3, 'mem_reg': 7 }
    LZCNT   = { 'reg_reg': 3, 'reg_mem': 3 }
    TZCNT   = { 'reg_reg': 3, 'reg_mem': 3 }
    POPCNT  = { 'reg_reg': 3, 'reg_mem': 3 } 
    BLSI    = { 'reg_reg': 1 }
    BLSMSK  = { 'reg_reg': 1 }
    BLSR    = { 'reg_reg': 1 }
    BEXTR   = { 'reg_reg': 2, 'mem_reg': 6 }
    BZHI    = { 'reg_reg': 1 }
    PDEP    = { 'reg_reg': 3, 'reg_mem': 3 }
    PEXT    = { 'reg_reg': 3, 'reg_mem': 3 }
    ANDN    = { 'reg_reg': 1 }

    # --- Flags & System ---
    SETZ    = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 3 }
    SETNZ   = { 'reg_reg': 1, 'reg_mem': 3 }
    SETO    = { 'reg_reg': 1, 'reg_mem': 3 }
    SETNLE  = { 'reg_reg': 1, 'reg_mem': 3 }
    CLC     = { 'reg_reg': 1 } 
    STC     = { 'reg_reg': 1 } 
    CMC     = { 'reg_reg': 1 } 
    LAHF    = { 'reg_reg': 4 } 
    SAHF    = { 'reg_reg': 1 }
    SYSCALL = { 'reg_reg': 40 }

    # --- Control Transfer ---
    JMP       = { 'reg_reg': 2, 'mem_reg': 2 }
    CALL      = { 'reg_reg': 3, 'mem_reg': 4 } 
    CALL_NEAR = { 'reg_reg': 2, 'mem_reg': 4, 'reg_mem': 2 } 
    RET       = { 'reg_reg': 2, 'mem_reg': 2, 'reg_mem': 2 }
    RET_NEAR  = { 'reg_reg': 2, 'mem_reg': 2, 'reg_mem': 2 }
    LOOP      = { 'reg_reg': 7 }
    JZ        = { 'reg_reg': 1 }
    JE        = { 'reg_reg': 1 }
    JNZ       = { 'reg_reg': 1 }
    JNE       = { 'reg_reg': 1 }
    JG        = { 'reg_reg': 1 }
    JGE       = { 'reg_reg': 1 }
    JL        = { 'reg_reg': 1 }
    JNL       = { 'reg_reg': 1 }
    JLE       = { 'reg_reg': 1 }
    JNLE      = { 'reg_reg': 1 }
    JA        = { 'reg_reg': 1 }
    JAE       = { 'reg_reg': 1 }
    JS        = { 'reg_reg': 1 }
    JNS       = { 'reg_reg': 1 }
    JB        = { 'reg_reg': 1 }
    JNB       = { 'reg_reg': 1 }
    JBE       = { 'reg_reg': 1 }
    JNBE      = { 'reg_reg': 1 }
    JECXZ     = { 'reg_reg': 2 }
    JRCXZ     = { 'reg_reg': 2 } 

    # --- Strings (Scalar) ---
    LODS    = { 'reg_reg': 3, 'reg_mem': 3 } 
    STOS    = { 'reg_reg': 3, 'reg_mem': 3 }
    STOSQ   = { 'reg_reg': 3, 'reg_mem': 3 }
    MOVS    = { 'reg_reg': 5 } 
    SCAS    = { 'reg_reg': 3, 'reg_mem': 3} 
    CMPS    = { 'reg_reg': 5 } 

    # --- Atomic/Sync ---
    XADD          = { 'reg_mem': 7 }
    CMPXCHG       = { 'reg_mem': 7 }
    LOCK_XADD     = { 'reg_mem': 21 }
    LOCK_ADD      = { 'reg_mem': 21 }
    CMPXCHG_LOCK  = { 'mem_reg': 22 }
    LFENCE        = { 'reg_reg': 5 } 
    MFENCE        = { 'reg_reg': 36 } 
    SFENCE        = { 'reg_reg': 6 } 
    PAUSE         = { 'reg_reg': 138 } 
    LEAVE         = { 'mem_reg': 4 }

    # --- Conversion (Scalar) ---
    CBW     = { 'reg_reg': 1 }
    CWDE    = { 'reg_reg': 1 }
    CDQE    = { 'reg_reg': 1 }
    CWD     = { 'reg_reg': 1 }
    CDQ     = { 'reg_reg': 1 }
    CQO     = { 'reg_reg': 1 }

    # --- Misc ---
    NOP     = { 'reg_reg': 0 } 
    CPUID   = { 'reg_reg': 150 } 
    RDTSC   = { 'reg_reg': 21 } 
    RDTSCP  = { 'reg_reg': 22 } 
    RDRAND  = { 'reg_reg': 2000 } 
    RDSEED  = { 'reg_reg': 2000 } 
    XGETBV  = { 'reg_reg': 8 }

    # =========================================================================
    #                        INTEGER VECTOR INSTRUCTIONS
    # =========================================================================
    
    # --- Vector Move ---
    MOVD      = { 'reg_reg': 2, 'mem_reg': 2, 'reg_mem': 2 } 
    MOVQ      = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 2 }
    MOVDQA    = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 3 } 
    MOVDQU    = { 'reg_reg': 1, 'mem_reg': 3, 'reg_mem': 3 } 
    VMOVD     = { 'reg_reg': 0, 'mem_reg': 3, 'reg_mem': 4 }
    VMOVDQA   = { 'reg_reg': 0, 'mem_reg': 3, 'reg_mem': 4 }
    VMOVDQU   = { 'reg_reg': 0, 'mem_reg': 3, 'reg_mem': 4 }
    LDDQU     = { 'mem_reg': 3 } 
    MOVNTQ    = { 'reg_mem': 450 } # Non-temporal
    MOVNTDQ   = { 'reg_mem': 450 } 
    MOVNTDQA  = { 'mem_reg': 5 } 
    PMOVMSKB  = { 'reg_reg': 2 }
    VPMOVMSKB = { 'reg_reg': 2 }

    # --- Pack / Unpack ---
    PACKSSWB   = { 'reg_reg': 1 } 
    PACKSSDW   = { 'reg_reg': 1 } 
    PACKUSWB   = { 'reg_reg': 1 } 
    PACKUSDW   = { 'reg_reg': 1 } 
    PUNPCKHBW  = { 'reg_reg': 1 } 
    PUNPCKLBW  = { 'reg_reg': 1 }
    PUNPCKLWD  = { 'reg_reg': 1 } 
    PUNPCKLDQ  = { 'reg_reg': 1 }
    PUNPCKLQDQ = { 'reg_reg': 1 }
    # Note: Other PUNPCK variants generally share latency 1

    # --- Shuffle / Permute ---
    PSHUFB    = { 'reg_reg': 1 } 
    PSHUFD    = { 'reg_reg': 1 } 
    PSHUFW    = { 'reg_reg': 1 } 
    PSHUFLW   = { 'reg_reg': 1 } 
    PSHUFHW   = { 'reg_reg': 1 } 
    PALIGNR   = { 'reg_reg': 1 } 
    VALIGND   = { 'reg_reg': 3, 'mem_reg': 3 } 
    VALIGNQ   = { 'reg_reg': 3, 'mem_reg': 3 }  
    VPERMD    = { 'reg_reg': 3, 'mem_reg': 3 } 
    VPERMQ    = { 'reg_reg': 3, 'mem_reg': 3 } 
    VPERM2I128= { 'reg_reg': 3, 'mem_reg': 3 } 

    # --- Insert / Extract ---
    PEXTRB    = { 'reg_reg': 3, 'reg_mem': 3 } 
    PEXTRW    = { 'reg_reg': 3, 'reg_mem': 3 } 
    PEXTRD    = { 'reg_reg': 3, 'reg_mem': 3 } 
    PEXTRQ    = { 'reg_reg': 3, 'reg_mem': 3 } 
    PINSRB    = { 'reg_reg': 3, 'mem_reg': 3 } 
    PINSRW    = { 'reg_reg': 3, 'mem_reg': 3 } 
    PINSRD    = { 'reg_reg': 3, 'mem_reg': 3 } 
    PINSRQ    = { 'reg_reg': 3, 'mem_reg': 3 } 
    VINSERTI128 = { 'reg_reg': 3, 'mem_reg': 5 } 
    VEXTRACTI128 = { 'reg_reg': 3, 'reg_mem': 4 } 

    # --- Broadcast / Gather / Scatter ---
    VPBROADCASTB = { 'reg_reg': 3, 'mem_reg': 3 } 
    VPBROADCASTD = { 'reg_reg': 3, 'mem_reg': 3 } 
    VPBROADCASTQ = { 'reg_reg': 3, 'mem_reg': 3 } 
    VPGATHERDD   = { 'mem_reg': 6 } # Estimate based on complex output
    VPGATHERDQ   = { 'mem_reg': 6 } 
    VPSCATTERDD  = { 'reg_mem': 6 } 
    VPSCATTERDQ  = { 'reg_mem': 5 } 

    # --- Vector Arithmetic ---
    PADDB     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDW     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDD     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDQ     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PSUBB     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PSUBW     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PSUBD     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PSUBQ     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDSB    = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDSW    = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDUSB   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PADDUSW   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PHADDW    = { 'reg_reg': 2 } 
    PHADDD    = { 'reg_reg': 2 } 
    PHSUBW    = { 'reg_reg': 2 } 
    PHSUBD    = { 'reg_reg': 2 } 
    PAVGB     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PAVGW     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PABSB     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PABSW     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PABSD     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PMINUB    = { 'reg_reg': 1, 'mem_reg': 5 } 
    VPMINUB   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PMAXUB    = { 'reg_reg': 1, 'mem_reg': 5 } 

    # --- Vector Multiply ---
    PMULLW    = { 'reg_reg': 5 } 
    PMULLD    = { 'reg_reg': 10 } 
    PMULHW    = { 'reg_reg': 5 } 
    PMULHUW   = { 'reg_reg': 5 } 
    PMULUDQ   = { 'reg_reg': 5 } 
    PMULDQ    = { 'reg_reg': 5 } 
    PMADDWD   = { 'reg_reg': 5 } 
    PMADDUBSW = { 'reg_reg': 5 } 
    VPDPBUSD  = { 'reg_reg': 5 } # AVX512 VNNI
    
    # --- Vector Compare ---
    PCMPEQB   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPEQW   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPEQD   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPEQQ   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPGTB   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPGTW   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPGTD   = { 'reg_reg': 1, 'mem_reg': 5 } 
    PCMPGTQ   = { 'reg_reg': 3, 'mem_reg': 3 }  
    VPCMPEQB  = { 'reg_reg': 3, 'mem_reg': 3 } 

    # --- Vector Logical ---
    PAND      = { 'reg_reg': 1, 'mem_reg': 5 } 
    PANDN     = { 'reg_reg': 1, 'mem_reg': 5 } 
    POR       = { 'reg_reg': 1, 'mem_reg': 5 } 
    PXOR      = { 'reg_reg': 1, 'mem_reg': 5 }
    VPOR      = { 'reg_reg': 1, 'mem_reg': 5 }
    VPXOR     = { 'reg_reg': 1, 'mem_reg': 5 } 
    PTEST     = { 'reg_reg': 3 } 
    VPTEST    = { 'reg_reg': 5 } 

    # --- Vector Shift ---
    PSLLW     = { 'reg_reg': 1 } 
    PSLLD     = { 'reg_reg': 1 } 
    PSLLQ     = { 'reg_reg': 1 } 
    PSRLW     = { 'reg_reg': 1 } 
    PSRLD     = { 'reg_reg': 1 } 
    PSRLQ     = { 'reg_reg': 1 } 
    PSRAW     = { 'reg_reg': 1 } 
    PSRAD     = { 'reg_reg': 1 } 
    PSLLDQ    = { 'reg_reg': 1 } 
    PSRLDQ    = { 'reg_reg': 1 } 

    # --- Encryption / String Support ---
    PCMPESTRI = { 'reg_reg': 11 } 
    PCMPESTRM = { 'reg_reg': 11 } 
    PCMPISTRI = { 'reg_reg': 10 } 
    PCMPISTRM = { 'reg_reg': 9 } 
    AESENC    = { 'reg_reg': 3 } 
    AESENCLAST= { 'reg_reg': 3 } 
    AESDEC    = { 'reg_reg': 3 } 
    AESDECLAST= { 'reg_reg': 3 } 
    PCLMULQDQ = { 'reg_reg': 6 } 
    SHA1RNDS4 = { 'reg_reg': 6 } 
    SHA256RNDS2 = { 'reg_reg': 6 }
    
    # =========================================================================
    #                   FLOATING POINT VECTOR INSTRUCTIONS
    # =========================================================================
    
    # --- Data Movement ---
    MOVAPS    = { 'reg_reg': 1, 'mem_reg': 4, 'reg_mem': 3 } # VMOVAPS
    MOVUPS    = { 'reg_reg': 1, 'mem_reg': 4, 'reg_mem': 3 }
    MOVSS     = { 'reg_reg': 1, 'mem_reg': 4, 'reg_mem': 3 }
    MOVSD     = { 'reg_reg': 1, 'mem_reg': 4, 'reg_mem': 3 }
    MOVHPS    = { 'mem_reg': 4, 'reg_mem': 3 }
    MOVHPD    = { 'mem_reg': 4, 'reg_mem': 3 }
    MOVLPS    = { 'mem_reg': 4, 'reg_mem': 3 }
    MOVLPD    = { 'mem_reg': 4, 'reg_mem': 3 }
    MOVHLPS   = { 'reg_reg': 1 }
    MOVLHPS   = { 'reg_reg': 1 }
    MOVNTPS   = { 'reg_mem': 450 }
    MOVNTPD   = { 'reg_mem': 450 }
    MOVDDUP   = { 'reg_reg': 1, 'mem_reg': 3 }
    MOVSHDUP  = { 'reg_reg': 1 }
    MOVSLDUP  = { 'reg_reg': 1 }

    # --- Shuffle / Permute / Blend ---
    SHUFPS    = { 'reg_reg': 1 }
    SHUFPD    = { 'reg_reg': 1 }
    VPERMILPS = { 'reg_reg': 1 }
    VPERMILPD = { 'reg_reg': 1 }
    VPERM2F128= { 'reg_reg': 3 }
    BLENDPS   = { 'reg_reg': 1 }
    BLENDPD   = { 'reg_reg': 1 }
    BLENDVPS  = { 'reg_reg': 1 }
    BLENDVPD  = { 'reg_reg': 1 }
    
    # --- Insert / Extract ---
    EXTRACTPS = { 'reg_reg': 3, 'reg_mem': 5 }
    INSERTPS  = { 'reg_reg': 1, 'mem_reg': 4 }
    VINSERTF128  = { 'reg_reg': 3, 'mem_reg': 5 }
    VEXTRACTF128 = { 'reg_reg': 3, 'reg_mem': 5 }

    # --- Broadcast / Gather / Scatter (Float) ---
    VBROADCASTSS = { 'reg_reg': 1, 'mem_reg': 3 }
    VBROADCASTSD = { 'reg_reg': 1, 'mem_reg': 3 }
    VGATHERDPS   = { 'mem_reg': 6 } 
    VGATHERQPS   = { 'mem_reg': 6 }
    VSCATTERDPS  = { 'reg_mem': 6 }

    # --- Arithmetic (Float) ---
    ADDSS     = { 'reg_reg': 4, 'mem_reg': 5 } # Latency 4
    ADDSD     = { 'reg_reg': 4, 'mem_reg': 5 }
    ADDPS     = { 'reg_reg': 4, 'mem_reg': 5 }
    ADDPD     = { 'reg_reg': 4, 'mem_reg': 5 }
    SUBSS     = { 'reg_reg': 4, 'mem_reg': 5 }
    SUBSD     = { 'reg_reg': 4, 'mem_reg': 5 }
    SUBPS     = { 'reg_reg': 4, 'mem_reg': 5 }
    SUBPD     = { 'reg_reg': 4, 'mem_reg': 5 }
    MULSS     = { 'reg_reg': 4, 'mem_reg': 5 }
    MULSD     = { 'reg_reg': 4, 'mem_reg': 5 }
    MULPS     = { 'reg_reg': 4, 'mem_reg': 5 }
    MULPD     = { 'reg_reg': 4, 'mem_reg': 5 }
    DIVSS     = { 'reg_reg': 11 } 
    DIVSD     = { 'reg_reg': 14 } # 13-14
    DIVPS     = { 'reg_reg': 11 }
    DIVPD     = { 'reg_reg': 14 } # 13-14
    SQRTSS    = { 'reg_reg': 12 }
    SQRTSD    = { 'reg_reg': 16 } # 15-16
    SQRTPS    = { 'reg_reg': 12 }
    SQRTPD    = { 'reg_reg': 16 }
    RSQRTSS   = { 'reg_reg': 4 }
    RSQRTPS   = { 'reg_reg': 4 }
    RCPSS     = { 'reg_reg': 4 }
    RCPPS     = { 'reg_reg': 4 }
    
    # --- FMA (Fused Multiply-Add) ---
    VFMADD132PS = { 'reg_reg': 4, 'mem_reg': 5 } 
    VFMADD213PS = { 'reg_reg': 4, 'mem_reg': 5 }
    VFMADD231PS = { 'reg_reg': 4, 'mem_reg': 5 }
    # (Note: All FMA variants generally share 4 cycle latency)

    # --- Comparison / Logic (Float) ---
    CMPSS     = { 'reg_reg': 3 }
    CMPSD     = { 'reg_reg': 3 }
    CMPPS     = { 'reg_reg': 3 }
    CMPPD     = { 'reg_reg': 3 }
    COMISS    = { 'reg_reg': 2 }
    COMISD    = { 'reg_reg': 2 }
    UCOMISS   = { 'reg_reg': 2 }
    UCOMISD   = { 'reg_reg': 2 }
    MAXSS     = { 'reg_reg': 4, 'mem_reg': 5 }
    MAXSD     = { 'reg_reg': 4, 'mem_reg': 5 }
    MAXPS     = { 'reg_reg': 4, 'mem_reg': 5 }
    MAXPD     = { 'reg_reg': 4, 'mem_reg': 5 }
    MINSS     = { 'reg_reg': 4, 'mem_reg': 5 }
    MINSD     = { 'reg_reg': 4, 'mem_reg': 5 }
    MINPS     = { 'reg_reg': 4, 'mem_reg': 5 }
    MINPD     = { 'reg_reg': 4, 'mem_reg': 5 }
    ANDPS     = { 'reg_reg': 1, 'mem_reg': 5 }
    ANDPD     = { 'reg_reg': 1, 'mem_reg': 5 }
    ORPS      = { 'reg_reg': 1, 'mem_reg': 5 }
    ORPD      = { 'reg_reg': 1, 'mem_reg': 5 }
    XORPS     = { 'reg_reg': 1, 'mem_reg': 5 }
    XORPD     = { 'reg_reg': 1, 'mem_reg': 5 }

    # --- Conversions (Float <-> Int/Float) ---
    CVTPD2PS  = { 'reg_reg': 4 } # Estimate (often slower than simple cast)
    CVTPS2PD  = { 'reg_reg': 4 } 
    CVTSS2SD  = { 'reg_reg': 5 }
    CVTSD2SS  = { 'reg_reg': 5 }
    CVTPS2DQ  = { 'reg_reg': 4 }
    CVTDQ2PS  = { 'reg_reg': 4 }
    CVTTPS2DQ = { 'reg_reg': 4 }
    CVTSS2SI  = { 'reg_reg': 6 }
    CVTSD2SI  = { 'reg_reg': 6 }
    CVTSI2SS  = { 'reg_reg': 5 } # Load/Op
    CVTSI2SD  = { 'reg_reg': 5 }

    # --- Math / Special ---
    ROUNDPS   = { 'reg_reg': 8 } 
    ROUNDPD   = { 'reg_reg': 8 }
    DPPS      = { 'reg_reg': 14 }
    DPPD      = { 'reg_reg': 9 }
    
    # --- Other ---
    VZEROUPPER  = { 'reg_reg': 1 }
    XRSTOR      = { 'mem_reg': 80 }
    XSAVE       = { 'mem_reg': 137 }
    XSAVEC      = { 'reg_mem': 84 }
    
    def reg_reg_latency(self):
        try:
            return self.value['reg_reg']
        except KeyError:
            raise ValueError(f"Opcode {self.name} does not have a reg_reg latency.")
            
    def reg_mem_latency(self):
        try:
            return self.value['reg_mem']
        except KeyError:
            raise ValueError(f"Opcode {self.name} does not have a reg_mem latency.")
            
    def mem_reg_latency(self):
        try:
            return self.value['mem_reg']
        except KeyError:
            raise ValueError(f"Opcode {self.name} does not have a mem_reg latency.")