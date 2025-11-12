# Peregrine Tracing Tool

The Peregrine tracing tool is a dynamic binary instrumentation tool built on Intel Pin that captures detailed execution traces of x86-64 programs. It records instruction-level information including register operations, memory accesses, dependencies, branch types, and synchronization barriers.

## Overview

The tracing tool instruments programs at runtime and generates a comprehensive CSV trace file containing:

- **Instruction Pointers (IP)**: Memory addresses of each instruction
- **Assembly Code**: Disassembled instruction text
- **Instruction Categories**: High-level category (e.g., BINARY, DATAXFER, CONTROL)
- **Opcodes**: Specific instruction opcodes
- **Branch Types**: Classification of branches (direct_unconditional, direct_conditional, indirect)
- **Synchronization Barriers**: Detection of memory fences, atomic operations, and serializing instructions
- **Register Operations**: Registers read and written by each instruction
- **Memory Operations**: Memory addresses read from and written to
- **Dependencies**: Register and memory dependencies between instructions

## Features

### Instruction Tracking
- Captures every executed instruction with its IP address
- Records assembly disassembly and instruction metadata
- Tracks instruction categories and opcodes

### Branch Analysis
- **Direct Unconditional Branches**: Direct jumps (e.g., `jmp label`) - tracked as "direct_unconditional"
- **Direct Conditional Branches**: Conditional jumps (e.g., `jne`, `je`) - tracked as "direct_conditional"
- **Indirect Branches**: Branches through registers or memory (e.g., `jmp *%rax`)

### Synchronization Barriers
Detects and marks synchronization points including:
- Memory fences: `MFENCE`, `SFENCE`, `LFENCE`
- Serializing instructions: `CPUID`, `SERIALIZE`
- Atomic operations: `XCHG`, `CMPXCHG`, `XADD`
- Cache operations: `CLFLUSH`, `CLFLUSHOPT`, `CLWB`
- Timestamp instructions: `RDTSC`, `RDTSCP`
- Transactional memory: `XBEGIN`, `XEND`, `XABORT`
- Interrupt returns: `IRET`, `IRETD`, `IRETQ`

### Dependency Tracking
- **Register Dependencies**: Tracks which instructions write registers that are later read
- **Memory Dependencies**: Tracks which instructions write memory addresses that are later read
- Dependencies are computed at execution time to ensure accurate ordering

### Memory Access Tracking
- Records all memory read and write addresses
- Tracks memory access sizes
- Handles overlapping memory writes correctly

## Installation

### Prerequisites
- Linux x86-64 system
- GCC compiler
- Python 3 (for verification script)

### Installing Intel Pin

1. **Download Intel Pin**:
   - Visit the [Intel Pin download page](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html)
   - Download the Linux IA32 and intel64 (x86 32 bit and 64 bit) version
   - For example, Pin 3.31: `pin-external-3.31-98869-gfa6f126a8-gcc-linux.tar.gz`

2. **Extract Pin**:
   ```bash
   tar -xzf pin-external-3.31-98869-gfa6f126a8-gcc-linux.tar.gz
   ```

3. **Set Environment Variables** (optional but recommended):
   ```bash
   export PIN_ROOT=/path/to/pin-external-3.31-98869-gfa6f126a8-gcc-linux
   export PATH=$PIN_ROOT:$PATH
   ```

### Building the Tracing Tool

1. **Copy the source file to Pin's tools directory**:
   ```bash
   cp peregrine-trace.cpp $PIN_ROOT/source/tools/MyPinTool/
   cd $PIN_ROOT/source/tools/MyPinTool/
   ```

2. **Build the pintool**:
   ```bash
   make obj-intel64/peregrine-trace.so
   ```

   This will create the shared object file `obj-intel64/peregrine-trace.so` that can be used with Pin.

## Usage

### Basic Usage

To trace a program, use Pin with the tracing tool:

```bash
$PIN_ROOT/pin -t obj-intel64/peregrine-trace.so -- <your_program> [program_args]
```

For example:
```bash
$PIN_ROOT/pin -t obj-intel64/peregrine-trace.so -- ls -la
```

### Output Files

The tracing tool generates two output files:

1. **`trace.csv`**: Main trace file containing instruction-by-instruction execution data
   - CSV format with columns: IP, Assembly, Category, Opcode, Branch Type, Instruction Sync, Read Registers, Write Registers, Register Dependent IPs, Read Addresses, Write Addresses, Memory Dependent IPs

2. **`trace_summary.txt`**: Summary statistics
   - Number of instruction sync points
   - Counts of different branch types

### CSV Output Format

Each row in `trace.csv` represents one executed instruction with the following fields:

- **IP**: Instruction pointer (hexadecimal address)
- **Assembly**: Disassembled instruction text
- **Category**: Instruction category (e.g., BINARY, DATAXFER, CONTROL)
- **Opcode**: Instruction opcode (e.g., MOV, ADD, JMP)
- **Branch Type**: Branch classification (empty for non-branches)
- **Instruction Sync**: "true" if instruction is a synchronization barrier, "false" otherwise
- **Read Registers**: Semicolon-separated list of registers read (e.g., "rax;rbx"). Partial registers (e.g., EAX, AX, AL) are normalized to their full register (e.g., RAX) for accurate dependency tracking.
- **Write Registers**: Semicolon-separated list of registers written. Partial registers are normalized to full registers.
- **Register Dependent IPs**: Semicolon-separated list of IPs of instructions that wrote registers read by this instruction
- **Read Addresses**: Semicolon-separated list of memory addresses read (format: `0xADDR(SIZE)`)
- **Write Addresses**: Semicolon-separated list of memory addresses written (format: `0xADDR(SIZE)`)
- **Memory Dependent IPs**: Semicolon-separated list of IPs of instructions that wrote to memory addresses read by this instruction

## Testing

### Test Files

The tracing directory includes test files:
- `test_trace_c.c`: C wrapper for the test function
- `test_trace_c.s`: Assembly test file with various instruction patterns
- `verify_trace.py`: Python script to verify trace correctness

### Building Test Programs

```bash
# Build the test program
gcc -o test_trace_c test_trace_c.c test_trace_c.s
```

### Running Tests

```bash
# Run the test program directly
./test_trace_c

# Run with Pin tracing
$PIN_ROOT/pin -t obj-intel64/peregrine-trace.so -- ./test_trace_c
```

### Verifying Traces

Use the verification script to check trace correctness:

```bash
python3 verify_trace.py trace.csv
```

The verification script checks:
- IP tracking for all instructions
- Branch type detection (direct_unconditional, direct_conditional, indirect)
- Synchronization barrier detection
- Register read/write tracking
- Register dependency tracking
- Memory read/write tracking
- Memory dependency tracking
- Test-specific patterns

For detailed test coverage information, see `TEST_README.md`.

## Example Output

### Trace CSV Sample

```csv
IP,Assembly,Category,Opcode,Branch Type,Instruction Sync,Read Registers,Write Registers,Register Dependent IPs,Read Addresses,Write Addresses,Memory Dependent IPs
0x400000,mov $0x100, %rax,DATAXFER,MOV,,false,,rax,,,
0x400007,mov %rax, %rbx,DATAXFER,MOV,,false,rax,rbx,0x400000,,
0x40000a,mov test_var(%rip), %rdx,DATAXFER,MOV,,false,,rdx,,0x600000(8),
0x400011,mfence,CONTROL,MFENCE,,true,,,,
```

### Trace Summary Sample

```
Number of instruction sync points: 5
Number of direct unconditional branches: 1
Number of direct conditional branches: 2
Number of indirect branches: 1
```

## Notes

- The tracing tool tracks dependencies at **execution time**, not instrumentation time, ensuring accurate dependency information even with out-of-order execution or optimizations.
- **Register normalization**: Partial registers (e.g., EAX, AX, AL, AH) are normalized to their full register (e.g., RAX) using `REG_FullRegName()` before tracking. This ensures that all operations on the same architectural register are correctly tracked for dependency analysis.
- Memory write tracking handles overlapping writes correctly by splitting memory ranges.
- The tool may significantly slow down program execution due to the overhead of instrumentation.
- Large programs may generate very large trace files.

## Troubleshooting

- **"ERROR: could not init pin..."**: Ensure Pin is properly installed and the path is correct
- **Missing dependencies in trace**: Some dependencies may not appear due to program structure (e.g., loops, indirect branches)
- **Large trace files**: Consider filtering or sampling for very long-running programs

## License

Please refer to Intel Pin's license terms when using this tool.

