# Assembly Test File for Tracing System Verification

This directory contains minimal assembly test files designed to verify all aspects of the tracing system.

## Files

- `test_trace_c.s` + `test_trace_c.c` - C-callable assembly test (compiles with gcc)
- `Makefile.test` - Makefile for building the test

## Test Coverage

The test file exercises the following tracing features:

### 1. IP (Instruction Pointer) Tracking
- **Automatic**: Every instruction has an IP that should be tracked

### 2. Register Read/Write
- **Test 1**: `mov $0x100, %rax` (write RAX)
- **Test 1**: `mov %rax, %rbx` (read RAX, write RBX)
- **Test 13**: Complex chain with ADD, SUB, IMUL operations

### 3. Register Dependencies
- **Test 1**: `mov %rax, %rbx` depends on previous `mov $0x100, %rax`
- **Test 13**: Chain: R12 → R13 → R14 → R15 (each depends on previous)
- **Test 14**: RBX written from memory, then used to write memory

### 4. Memory Read Operations
- **Test 2**: `mov test_var(%rip), %rdx` (read from memory)
- **Test 4**: `mov test_array(%rip), %r8` (read from memory)
- **Test 14**: Multiple memory reads

### 5. Memory Write Operations
- **Test 3**: `mov %rsi, test_var(%rip)` (write to memory)
- **Test 4**: `mov %rdi, test_array(%rip)` (write to memory)
- **Test 14**: Multiple memory writes

### 6. Memory Dependencies
- **Test 4**: Write to `test_array[0]`, then read from `test_array[0]` (memory dependency)
- **Test 14**: Write to `test_array[1]`, read from `test_array[1]`, write to `test_array[2]`, read from `test_array[2]`

### 7. Branch Types

#### Direct Unconditional Branch
- **Test 5**: `jmp branch_label1` (direct unconditional)
- **Test 23**: `jmp test_uncond_branch1` (direct unconditional, with dummy instructions for IP difference detection)
- **Test 24**: `jmp test_uncond_branch2` (direct unconditional, another example with dummy instructions)

#### Direct Conditional Branch (Taken)
- **Test 6**: `cmp $0, %rax; jne branch_label2` (direct conditional, will be taken since RAX != 0)
- **Test 25**: `cmp $0, %rdx; jne test_cond_taken1` (direct conditional JNE, will be taken since RDX != 0, with dummy instructions)
- **Test 26**: `cmp $0x30, %r10; jg test_cond_taken2` (direct conditional JG, will be taken since 80 > 48, with dummy instructions)
- **Test 29**: `test %rdx, %rdx; jz test_cond_taken3` (direct conditional JZ, will be taken since RDX == 0 sets ZF, with dummy instructions)

#### Direct Conditional Branch (Not Taken)
- **Test 7**: `cmp $0, %rax; je branch_label3` (direct conditional, will NOT be taken since RAX != 0)
- **Test 27**: `cmp $0, %r13; je test_cond_not_taken1` (direct conditional JE, will NOT be taken since R13 != 0, with dummy instructions)
- **Test 28**: `cmp $0x30, %rax; jl test_cond_not_taken2` (direct conditional JL, will NOT be taken since 80 is NOT < 48, with dummy instructions)
- **Test 30**: `test %rdi, %rdi; jz test_cond_not_taken3` (direct conditional JZ, will NOT be taken since RDI != 0, ZF not set, with dummy instructions)

#### Indirect Branch
- **Test 8**: `lea branch_label4(%rip), %r11; jmp *%r11` (indirect branch via register)
- **Test 31**: `lea test_indirect_target1(%rip), %r10; jmp *%r10` (indirect branch via register R10, with dummy instructions)
- **Test 32**: `lea test_indirect_target2(%rip), %r12; mov %r12, %r13; jmp *%r13` (indirect branch via register R13, with dummy instructions)
- **Test 33**: `lea test_indirect_target3(%rip), %r15; mov %r15, test_var(%rip); jmp *test_var(%rip)` (indirect branch via memory, with dummy instructions)
- **Test 34**: `lea test_indirect_target4(%rip), %rcx; add $0, %rcx; jmp *%rcx` (indirect branch via register RCX with dependency chain, with dummy instructions)

**Note**: Tests 23-34 include dummy instructions (nop, mov, add, sub) between branch instructions and their target labels to create significant IP differences that can be easily detected in the trace. This helps verify that branch target addresses are correctly identified and IP jumps are properly tracked.

### 8. Instruction Sync Barriers

#### Memory Fences
- **Test 9**: `mfence` (memory fence)
- **Test 9**: `sfence` (store fence)
- **Test 9**: `lfence` (load fence)

#### CPUID
- **Test 10**: `cpuid` (serializing instruction)

#### Atomic Operations
- **Test 11**: `xchg %rax, test_var(%rip)` (atomic exchange, sync barrier)
- **Test 15**: `cmpxchg %rbx, test_var(%rip)` (compare and exchange, sync barrier)

#### Timestamp Instructions
- **Test 12**: `rdtsc` (read timestamp counter, sync barrier)
- **Test 16**: `rdtscp` (read timestamp counter with processor ID, sync barrier)

### 9. Register Normalization

#### 32-bit Partial Registers
- **Test 17**: `mov $0x12345678, %eax` (write EAX, should normalize to RAX)
- **Test 17**: `mov $0x9ABCDEF0, %ebx` (write EBX, should normalize to RBX)
- **Test 17**: `mov %eax, %ecx` (read EAX, write ECX, should show reg dependency on previous EAX write)
- **Test 17**: `mov %ebx, %edx` (read EBX, write EDX, should show reg dependency on previous EBX write)

#### 16-bit Partial Registers
- **Test 18**: `mov $0x1111, %ax` (write AX, should normalize to RAX)
- **Test 18**: `mov $0x2222, %bx` (write BX, should normalize to RBX)
- **Test 18**: `mov %ax, %cx` (read AX, write CX, should show reg dependency on previous AX write)
- **Test 18**: `mov %bx, %dx` (read BX, write DX, should show reg dependency on previous BX write)

#### 8-bit Partial Registers
- **Test 19**: `mov $0xAA, %al` (write AL, should normalize to RAX)
- **Test 19**: `mov $0xBB, %ah` (write AH, should normalize to RAX)
- **Test 19**: `mov $0xCC, %bl` (write BL, should normalize to RBX)
- **Test 19**: `mov $0xDD, %bh` (write BH, should normalize to RBX)
- **Test 19**: `mov %al, %cl` (read AL, write CL, should show reg dependency on previous AL write)
- **Test 19**: `mov %ah, %ch` (read AH, write CH, should show reg dependency on previous AH write)

#### Mixed Partial and Full Registers
- **Test 20**: `mov $0xDEADBEEF, %eax` then `mov %rax, %rbx` (EAX write, then RAX read - should show dependency)
- **Test 20**: `mov $0xCAFEBABE, %rax` then `mov %eax, %ecx` (RAX write, then EAX read - should show dependency)

#### Extended 32-bit Registers
- **Test 21**: `mov $0x11111111, %esi` (write ESI, should normalize to RSI)
- **Test 21**: `mov $0x22222222, %edi` (write EDI, should normalize to RDI)
- **Test 21**: `mov %esi, %r8d` (read ESI, write R8D, should show reg dependency, zero-extends to R8)
- **Test 21**: `mov %edi, %r9d` (read EDI, write R9D, should show reg dependency, zero-extends to R9)

#### Extended 16-bit Registers
- **Test 22**: `mov $0x3333, %si` (write SI, should normalize to RSI)
- **Test 22**: `mov $0x4444, %di` (write DI, should normalize to RDI)
- **Test 22**: `mov %si, %r10w` (read SI, write R10W, should show reg dependency, zero-extends to R10)
- **Test 22**: `mov %di, %r11w` (read DI, write R11W, should show reg dependency, zero-extends to R11)

## Building and Running

```bash
# Build
make -f Makefile.test

# Or build manually
gcc -o test_trace_c test_trace_c.c test_trace_c.s

# Run directly
./test_trace_c

# Run with Pin tool
/path/to/pin -t obj-intel64/peregrine-trace.so -- ./test_trace_c
```

## Expected Trace Output

The trace should show:

1. **IP addresses** for every instruction
2. **Branch types**: "direct_unconditional", "direct_conditional", "indirect"
3. **Branch behavior**: Direct conditional branches should show whether they are taken or not taken based on the condition
4. **IP differences**: Significant IP gaps between branch instructions and their target addresses (especially in Tests 23-34 with dummy instructions)
5. **Instruction sync**: "true" for MFENCE, SFENCE, LFENCE, CPUID, XCHG, CMPXCHG, RDTSC, RDTSCP
4. **Read registers**: e.g., "rax", "rbx", etc. (partial registers like EAX, AX, AL are normalized to their full register, e.g., RAX)
5. **Write registers**: e.g., "rax", "rbx", etc. (partial registers are normalized to full registers for accurate dependency tracking)
6. **Register dependent IPs**: IPs of instructions that wrote registers being read
7. **Read addresses**: Memory addresses being read (format: `0xADDR(SIZE)`)
8. **Write addresses**: Memory addresses being written (format: `0xADDR(SIZE)`)
9. **Memory dependent IPs**: IPs of instructions that wrote to memory addresses being read

## Verification

### Automated Verification Script

A Python script (`verify_trace.py`) is provided to automatically verify the trace.csv file:

```bash
# Verify trace.csv
python3 verify_trace.py trace.csv

# Or use the Makefile target
make -f Makefile.test verify
```

The verification script checks:

1. **IP Tracking**: All instructions have IP addresses
2. **Branch Types**: Direct_unconditional, direct_conditional, and indirect branches (Tests 5-8, 23-34)
3. **Branch Behavior**: Direct conditional branches correctly identified as taken or not taken (Tests 6-7, 25-30)
4. **IP Differences**: Detectable IP gaps between branch instructions and target addresses (Tests 23-34 with dummy instructions)
5. **Sync Barriers**: MFENCE, SFENCE, LFENCE, CPUID, XCHG, CMPXCHG, RDTSC, RDTSCP are marked as sync
4. **Register Operations**: Register reads and writes are tracked
5. **Register Dependencies**: Instructions show dependencies on previous instructions that wrote the registers they read
6. **Register Normalization**: Partial registers (EAX, AX, AL, AH, ESI, SI, etc.) are normalized to full registers (RAX, RSI, etc.) for accurate dependency tracking
7. **Memory Operations**: Memory reads and writes are tracked with addresses
8. **Memory Dependencies**: Instructions show dependencies on previous instructions that wrote to memory addresses they read
9. **Test Patterns**: Specific patterns from the test assembly file are detected

The script will report:
- ✓ **Success**: Feature is working correctly
- ⚠ **Warnings**: Potential issues (e.g., missing dependencies that may be expected)
- ✗ **Errors**: Critical issues (e.g., sync barriers not marked correctly)

### Manual Verification Checklist

- [ ] All instructions have IP addresses
- [ ] Direct unconditional branch detected (jmp) - Tests 5, 23, 24
- [ ] Direct conditional branches detected (jne, je, jg, jl, jz) - Tests 6, 7, 25-30
- [ ] Direct conditional branches correctly identified as taken (jne, jg, jz when conditions are met) - Tests 6, 25, 26, 29
- [ ] Direct conditional branches correctly identified as not taken (je, jl, jz when conditions are not met) - Tests 7, 27, 28, 30
- [ ] Indirect branch detected (jmp *%reg, jmp *mem) - Tests 8, 31-34
- [ ] IP differences between branch instructions and target addresses are detectable (dummy instructions create gaps) - Tests 23-34
- [ ] Sync barriers detected (mfence, sfence, lfence, cpuid, xchg, cmpxchg, rdtsc, rdtscp)
- [ ] Register reads tracked (rax, rbx, rcx, etc.)
- [ ] Register writes tracked (rax, rbx, rcx, etc.)
- [ ] Register dependencies tracked (instruction reading RAX shows dependency on instruction that wrote RAX)
- [ ] Register normalization: 32-bit partial registers (EAX, EBX) normalize to full registers (RAX, RBX)
- [ ] Register normalization: 16-bit partial registers (AX, BX) normalize to full registers (RAX, RBX)
- [ ] Register normalization: 8-bit partial registers (AL, AH, BL, BH) normalize to full registers (RAX, RBX)
- [ ] Register normalization: Mixed partial and full registers show correct dependencies (EAX write → RAX read, RAX write → EAX read)
- [ ] Register normalization: Extended registers (ESI, EDI, SI, DI) normalize correctly (ESI→RSI, EDI→RDI, SI→RSI, DI→RDI)
- [ ] Memory reads tracked (test_var, test_array addresses)
- [ ] Memory writes tracked (test_var, test_array addresses)
- [ ] Memory dependencies tracked (instruction reading from test_array[0] shows dependency on instruction that wrote to test_array[0])

