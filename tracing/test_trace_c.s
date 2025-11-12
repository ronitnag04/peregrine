# C-callable version of the assembly test
# This version can be easily compiled with gcc and called from C

.section .data
    # Test data section
    test_var: .quad 0x1234567890ABCDEF
    test_array: .quad 0x1111111111111111, 0x2222222222222222, 0x3333333333333333

.section .text
.global test_trace_function
test_trace_function:
    push %rbp
    mov %rsp, %rbp
    
    # Test 1: Register read/write and register dependencies
    # Write to RAX, then read from RAX (register dependency)
    mov $0x100, %rax          # Write RAX
    mov %rax, %rbx            # Read RAX, Write RBX (reg dependency on previous)
    mov %rbx, %rcx            # Read RBX, Write RCX (reg dependency)
    
    # Test 2: Memory read operations
    mov test_var(%rip), %rdx  # Read from memory, Write RDX
    
    # Test 3: Memory write operations
    mov $0xDEADBEEF, %rsi
    mov %rsi, test_var(%rip)  # Write to memory (read RSI, write memory)
    
    # Test 4: Memory dependencies
    # Write to memory, then read from same location (memory dependency)
    mov $0xCAFEBABE, %rdi
    mov %rdi, test_array(%rip)  # Write to test_array[0]
    mov test_array(%rip), %r8   # Read from test_array[0] (mem dependency)
    
    # Test 5: Direct unconditional branch
    jmp branch_label1
    
branch_label1:
    # Test 6: Direct conditional branch (taken)
    cmp $0, %rax              # Compare (reads RAX)
    jne branch_label2         # Direct conditional branch (will be taken)
    
    # This won't execute
    mov $0x999, %r9
    
branch_label2:
    # Test 7: Direct conditional branch (not taken)
    cmp $0, %rax              # Compare (reads RAX)
    je branch_label3          # Direct conditional branch (will NOT be taken)
    
    # This will execute
    mov $0x888, %r10
    
branch_label3:
    # Test 8: Indirect branch
    lea branch_label4(%rip), %r11
    jmp *%r11                 # Indirect branch via register
    
branch_label4:
    # Test 9: Instruction sync barriers
    mfence                    # Memory fence (sync barrier)
    sfence                    # Store fence (sync barrier)
    lfence                    # Load fence (sync barrier)
    
    # Test 10: More sync barriers
    cpuid                     # CPUID (sync barrier)
    
    # Test 11: Atomic operations (sync barriers)
    mov $0x5555, %rax
    xchg %rax, test_var(%rip) # XCHG (sync barrier, reads/writes RAX and memory)
    
    # Test 12: RDTSC (sync barrier)
    rdtsc                     # Read timestamp counter (sync barrier)
    
    # Test 13: Complex register dependencies chain
    mov $1, %r12              # Write R12
    add %r12, %r13            # Read R12, Write R13 (reg dependency)
    sub %r13, %r14            # Read R13, Write R14 (reg dependency)
    or %r14, %r15             # Read R14, Write R15 (reg dependency)
    
    # Test 14: Multiple memory operations with dependencies
    mov $0xAAAA, %rax
    mov %rax, test_array+8(%rip)  # Write to test_array[1]
    mov test_array+8(%rip), %rbx   # Read from test_array[1] (mem dependency)
    mov %rbx, test_array+16(%rip)  # Write to test_array[2] (reg dependency on RBX)
    mov test_array+16(%rip), %rcx  # Read from test_array[2] (mem dependency)
    
    # Test 15: CMPXCHG (sync barrier, complex operation)
    mov $0x1234, %rax
    mov $0x5678, %rbx
    cmpxchg %rbx, test_var(%rip)  # CMPXCHG (sync barrier, reads/writes RAX, reads RBX, reads/writes memory)
    
    # Test 16: RDTSCP (sync barrier)
    rdtscp                    # Read timestamp counter with processor ID (sync barrier)
    
    # Test 17: Register normalization - 32-bit partial registers (EAX, EBX, etc.)
    # Write to 32-bit partial registers, should normalize to RAX, RBX, etc.
    mov $0x12345678, %eax     # Write EAX (should normalize to RAX)
    mov $0x9ABCDEF0, %ebx     # Write EBX (should normalize to RBX)
    mov %eax, %ecx            # Read EAX, Write ECX (should show reg dependency on previous EAX write)
    mov %ebx, %edx            # Read EBX, Write EDX (should show reg dependency on previous EBX write)
    
    # Test 18: Register normalization - 16-bit partial registers (AX, BX, etc.)
    # Write to 16-bit partial registers, should normalize to RAX, RBX, etc.
    mov $0x1111, %ax          # Write AX (should normalize to RAX)
    mov $0x2222, %bx          # Write BX (should normalize to RBX)
    mov %ax, %cx              # Read AX, Write CX (should show reg dependency on previous AX write)
    mov %bx, %dx              # Read BX, Write DX (should show reg dependency on previous BX write)
    
    # Test 19: Register normalization - 8-bit partial registers (AL, AH, BL, etc.)
    # Write to 8-bit partial registers, should normalize to RAX, RBX, etc.
    mov $0xAA, %al            # Write AL (should normalize to RAX)
    mov $0xBB, %ah            # Write AH (should normalize to RAX)
    mov $0xCC, %bl            # Write BL (should normalize to RBX)
    mov $0xDD, %bh            # Write BH (should normalize to RBX)
    mov %al, %cl              # Read AL, Write CL (should show reg dependency on previous AL write)
    mov %ah, %ch              # Read AH, Write CH (should show reg dependency on previous AH write)
    
    # Test 20: Register normalization - Mixed partial and full registers
    # Test that partial and full registers are treated as the same register for dependencies
    mov $0xDEADBEEF, %eax     # Write EAX (normalizes to RAX)
    mov %rax, %rbx            # Read RAX, Write RBX (should show reg dependency on previous EAX write)
    mov $0xCAFEBABE, %rax     # Write RAX
    mov %eax, %ecx            # Read EAX, Write ECX (should show reg dependency on previous RAX write)
    
    # Test 21: Register normalization - Extended registers (ESI, EDI)
    mov $0x11111111, %esi     # Write ESI (should normalize to RSI)
    mov $0x22222222, %edi     # Write EDI (should normalize to RDI)
    mov %esi, %r8d            # Read ESI, Write R8D (should show reg dependency, zero-extends to R8)
    mov %edi, %r9d            # Read EDI, Write R9D (should show reg dependency, zero-extends to R9)
    
    # Test 22: Register normalization - 16-bit extended registers (SI, DI)
    mov $0x3333, %si          # Write SI (should normalize to RSI)
    mov $0x4444, %di          # Write DI (should normalize to RDI)
    mov %si, %r10w            # Read SI, Write R10W (should show reg dependency, zero-extends to R10)
    mov %di, %r11w            # Read DI, Write R11W (should show reg dependency, zero-extends to R11)
    
    # Return
    pop %rbp
    ret

