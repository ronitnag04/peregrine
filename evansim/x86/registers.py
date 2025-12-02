from enum import IntEnum, unique

@unique
class Register(IntEnum):
    """
    Comprehensive Enum of x86/x64 registers.
    Values are assigned uniquely to allow distinction between 
    register widths (e.g., al vs rax).
    """

    # --- 64-bit General Purpose ---
    rax = 0
    rcx = 1
    rdx = 2
    rbx = 3
    rsp = 4
    rbp = 5
    rsi = 6
    rdi = 7
    r8  = 8
    r9  = 9
    r10 = 10
    r11 = 11
    r12 = 12
    r13 = 13
    r14 = 14
    r15 = 15

    # --- 32-bit General Purpose ---
    eax = 16
    ecx = 17
    edx = 18
    ebx = 19
    esp = 20
    ebp = 21
    esi = 22
    edi = 23
    r8d = 24
    r9d = 25
    r10d = 26
    r11d = 27
    r12d = 28
    r13d = 29
    r14d = 30
    r15d = 31

    # --- 16-bit General Purpose ---
    ax = 32
    cx = 33
    dx = 34
    bx = 35
    sp = 36
    bp = 37
    si = 38
    di = 39
    r8w = 40
    r9w = 41
    r10w = 42
    r11w = 43
    r12w = 44
    r13w = 45
    r14w = 46
    r15w = 47

    # --- 8-bit General Purpose (Low) ---
    al = 48
    cl = 49
    dl = 50
    bl = 51
    spl = 52
    bpl = 53
    sil = 54
    dil = 55
    r8b = 56
    r9b = 57
    r10b = 58
    r11b = 59
    r12b = 60
    r13b = 61
    r14b = 62
    r15b = 63

    # --- 8-bit General Purpose (High - Legacy) ---
    ah = 64
    ch = 65
    dh = 66
    bh = 67

    # --- Instruction Pointers ---
    rip = 68  # 64-bit
    eip = 69  # 32-bit
    ip  = 70  # 16-bit

    # --- Segment Registers ---
    es = 71
    cs = 72
    ss = 73
    ds = 74
    fs = 75
    gs = 76

    # --- Flags ---
    rflags = 77 # 64-bit
    eflags = 78 # 32-bit
    flags  = 79 # 16-bit

    # --- Control Registers ---
    cr0 = 80
    cr1 = 81
    cr2 = 82
    cr3 = 83
    cr4 = 84
    cr8 = 85  # 64-bit only

    # --- Debug Registers ---
    dr0 = 90
    dr1 = 91
    dr2 = 92
    dr3 = 93
    dr6 = 94
    dr7 = 95

    # --- Memory Management / System ---
    gdtr = 100
    ldtr = 101
    idtr = 102
    tr   = 103
    
    # --- x87 FPU (Stack) ---
    st0 = 110
    st1 = 111
    st2 = 112
    st3 = 113
    st4 = 114
    st5 = 115
    st6 = 116
    st7 = 117

    # --- MMX (Aliased to x87, but given unique IDs for representation) ---
    mm0 = 120
    mm1 = 121
    mm2 = 122
    mm3 = 123
    mm4 = 124
    mm5 = 125
    mm6 = 126
    mm7 = 127

    # --- XMM Registers (SSE - 128 bit) ---
    xmm0  = 130; xmm1  = 131; xmm2  = 132; xmm3  = 133
    xmm4  = 134; xmm5  = 135; xmm6  = 136; xmm7  = 137
    xmm8  = 138; xmm9  = 139; xmm10 = 140; xmm11 = 141
    xmm12 = 142; xmm13 = 143; xmm14 = 144; xmm15 = 145
    # AVX-512 extended XMMs
    xmm16 = 146; xmm17 = 147; xmm18 = 148; xmm19 = 149
    xmm20 = 150; xmm21 = 151; xmm22 = 152; xmm23 = 153
    xmm24 = 154; xmm25 = 155; xmm26 = 156; xmm27 = 157
    xmm28 = 158; xmm29 = 159; xmm30 = 160; xmm31 = 161

    # --- YMM Registers (AVX - 256 bit) ---
    ymm0  = 170; ymm1  = 171; ymm2  = 172; ymm3  = 173
    ymm4  = 174; ymm5  = 175; ymm6  = 176; ymm7  = 177
    ymm8  = 178; ymm9  = 179; ymm10 = 180; ymm11 = 181
    ymm12 = 182; ymm13 = 183; ymm14 = 184; ymm15 = 185
    ymm16 = 186; ymm17 = 187; ymm18 = 188; ymm19 = 189
    ymm20 = 190; ymm21 = 191; ymm22 = 192; ymm23 = 193
    ymm24 = 194; ymm25 = 195; ymm26 = 196; ymm27 = 197
    ymm28 = 198; ymm29 = 199; ymm30 = 200; ymm31 = 201

    # --- ZMM Registers (AVX-512 - 512 bit) ---
    zmm0  = 210; zmm1  = 211; zmm2  = 212; zmm3  = 213
    zmm4  = 214; zmm5  = 215; zmm6  = 216; zmm7  = 217
    zmm8  = 218; zmm9  = 219; zmm10 = 220; zmm11 = 221
    zmm12 = 222; zmm13 = 223; zmm14 = 224; zmm15 = 225
    zmm16 = 226; zmm17 = 227; zmm18 = 228; zmm19 = 229
    zmm20 = 230; zmm21 = 231; zmm22 = 232; zmm23 = 233
    zmm24 = 234; zmm25 = 235; zmm26 = 236; zmm27 = 237
    zmm28 = 238; zmm29 = 239; zmm30 = 240; zmm31 = 241

    # --- Bounds Registers (MPX - Deprecated but included for completeness) ---
    bnd0 = 250
    bnd1 = 251
    bnd2 = 252
    bnd3 = 253