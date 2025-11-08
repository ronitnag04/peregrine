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

#### Direct Conditional Branch (Taken)
- **Test 6**: `cmp $0, %rax; jne branch_label2` (direct conditional, will be taken since RAX != 0)

#### Direct Conditional Branch (Not Taken)
- **Test 7**: `cmp $0, %rax; je branch_label3` (direct conditional, will NOT be taken since RAX != 0)

#### Indirect Branch
- **Test 8**: `lea branch_label4(%rip), %r11; jmp *%r11` (indirect branch via register)

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
2. **Branch types**: "direct unconditional", "direct conditional", "indirect"
3. **Instruction sync**: "true" for MFENCE, SFENCE, LFENCE, CPUID, XCHG, CMPXCHG, RDTSC, RDTSCP
4. **Read registers**: e.g., "rax", "rbx", etc.
5. **Write registers**: e.g., "rax", "rbx", etc.
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
2. **Branch Types**: Direct unconditional, direct conditional, and indirect branches
3. **Sync Barriers**: MFENCE, SFENCE, LFENCE, CPUID, XCHG, CMPXCHG, RDTSC, RDTSCP are marked as sync
4. **Register Operations**: Register reads and writes are tracked
5. **Register Dependencies**: Instructions show dependencies on previous instructions that wrote the registers they read
6. **Memory Operations**: Memory reads and writes are tracked with addresses
7. **Memory Dependencies**: Instructions show dependencies on previous instructions that wrote to memory addresses they read
8. **Test Patterns**: Specific patterns from the test assembly file are detected

The script will report:
- ✓ **Success**: Feature is working correctly
- ⚠ **Warnings**: Potential issues (e.g., missing dependencies that may be expected)
- ✗ **Errors**: Critical issues (e.g., sync barriers not marked correctly)

### Manual Verification Checklist

- [ ] All instructions have IP addresses
- [ ] Direct unconditional branch detected (jmp)
- [ ] Direct conditional branches detected (jne, je)
- [ ] Indirect branch detected (jmp *%r11)
- [ ] Sync barriers detected (mfence, sfence, lfence, cpuid, xchg, cmpxchg, rdtsc, rdtscp)
- [ ] Register reads tracked (rax, rbx, rcx, etc.)
- [ ] Register writes tracked (rax, rbx, rcx, etc.)
- [ ] Register dependencies tracked (instruction reading RAX shows dependency on instruction that wrote RAX)
- [ ] Memory reads tracked (test_var, test_array addresses)
- [ ] Memory writes tracked (test_var, test_array addresses)
- [ ] Memory dependencies tracked (instruction reading from test_array[0] shows dependency on instruction that wrote to test_array[0])

