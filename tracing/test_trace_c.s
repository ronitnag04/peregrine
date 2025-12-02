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
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x1111, %rax
    mov $0x2222, %rbx
    add $1, %rax
    add $1, %rbx
    nop
    nop
    
branch_label1:
    # Test 6: Direct conditional branch (taken)
    cmp $0, %rax              # Compare (reads RAX)
    jne branch_label2         # Direct conditional branch (will be taken)
    
    # This won't execute
    mov $0x999, %r9
    
    # Dummy instructions to create IP difference (won't execute if branch is taken)
    nop
    nop
    mov $0x3333, %r10
    mov $0x4444, %r11
    add $1, %r10
    add $1, %r11
    nop
    nop
    
branch_label2:
    # Test 7: Direct conditional branch (not taken)
    cmp $0, %rax              # Compare (reads RAX)
    je branch_label3          # Direct conditional branch (will NOT be taken)
    
    # This will execute
    mov $0x888, %r10
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x5555, %r11
    mov $0x6666, %r12
    add $1, %r11
    add $1, %r12
    nop
    nop
    
branch_label3:
    # Test 8: Indirect branch
    lea branch_label4(%rip), %r11
    jmp *%r11                 # Indirect branch via register
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x7777, %r12
    mov $0x8888, %r13
    add $1, %r12
    add $1, %r13
    nop
    nop
    
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
    
    # ============================================
    # Comprehensive Branch Tests
    # ============================================
    
    # Test 23: Direct unconditional branch (JMP)
    mov $0x1111, %rax          # mov rax, 0x1111
    jmp test_uncond_branch1    # Direct unconditional branch
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0xAAAA, %rbx
    mov $0xBBBB, %rcx
    add $1, %rbx
    add $1, %rcx
    sub $1, %rbx
    sub $1, %rcx
    nop
    nop
    
test_uncond_branch1:
    mov $0x2222, %rbx
    
    # Test 24: Direct unconditional branch (JMP) - another example
    jmp test_uncond_branch2    # Direct unconditional branch
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0xCCCC, %rdx
    mov $0xDDDD, %rsi
    add $1, %rdx
    add $1, %rsi
    sub $1, %rdx
    sub $1, %rsi
    nop
    nop
    
test_uncond_branch2:
    mov $0x3333, %rcx
    
    # Test 25: Direct conditional branch - TAKEN (JNE)
    mov $0x100, %rdx           # Set RDX to non-zero
    cmp $0, %rdx               # Compare RDX with 0
    jne test_cond_taken1       # Direct conditional branch - WILL BE TAKEN (RDX != 0)
    
    # This code should NOT execute
    mov $0xDEAD, %r8
    
    # Dummy instructions to create IP difference (won't execute if branch is taken)
    nop
    nop
    mov $0xEEEE, %r9
    mov $0xFFFF, %r10
    add $1, %r9
    add $1, %r10
    sub $1, %r9
    sub $1, %r10
    nop
    nop
    
test_cond_taken1:
    mov $0x4444, %r9           # This will execute
    
    # Test 26: Direct conditional branch - TAKEN (JG - greater than)
    mov $0x50, %r10            # Set R10 to 80
    cmp $0x30, %r10            # Compare R10 with 48
    jg test_cond_taken2        # Direct conditional branch - WILL BE TAKEN (80 > 48)
    
    # This code should NOT execute
    mov $0xBEEF, %r11
    
    # Dummy instructions to create IP difference (won't execute if branch is taken)
    nop
    nop
    mov $0x1111, %r12
    mov $0x2222, %r13
    add $1, %r12
    add $1, %r13
    sub $1, %r12
    sub $1, %r13
    nop
    nop
    
test_cond_taken2:
    mov $0x5555, %r12          # This will execute
    
    # Test 27: Direct conditional branch - NOT TAKEN (JE - equal)
    mov $0x200, %r13           # Set R13 to non-zero
    cmp $0, %r13               # Compare R13 with 0
    je test_cond_not_taken1    # Direct conditional branch - WILL NOT BE TAKEN (R13 != 0)
    
    # This code WILL execute
    mov $0x6666, %r14          # This will execute
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x3333, %r15
    mov $0x4444, %rax
    add $1, %r15
    add $1, %rax
    sub $1, %r15
    sub $1, %rax
    nop
    nop
    
test_cond_not_taken1:
    mov $0x7777, %r15          # This will also execute
    
    # Test 28: Direct conditional branch - NOT TAKEN (JL - less than)
    mov $0x50, %rax            # Set RAX to 80
    cmp $0x30, %rax            # Compare RAX with 48
    jl test_cond_not_taken2    # Direct conditional branch - WILL NOT BE TAKEN (80 is NOT < 48)
    
    # This code WILL execute
    mov $0x8888, %rbx          # This will execute
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x5555, %rcx
    mov $0x6666, %rdx
    add $1, %rcx
    add $1, %rdx
    sub $1, %rcx
    sub $1, %rdx
    nop
    nop
    
test_cond_not_taken2:
    mov $0x9999, %rcx          # This will also execute
    
    # Test 29: Direct conditional branch - TAKEN (JZ - zero flag)
    xor %rdx, %rdx             # Set RDX to 0 (sets ZF)
    test %rdx, %rdx            # Test RDX (ZF will be set)
    jz test_cond_taken3        # Direct conditional branch - WILL BE TAKEN (ZF is set)
    
    # This code should NOT execute
    mov $0xAAAA, %rdx
    
    # Dummy instructions to create IP difference (won't execute if branch is taken)
    nop
    nop
    mov $0x7777, %rsi
    mov $0x8888, %rdi
    add $1, %rsi
    add $1, %rdi
    sub $1, %rsi
    sub $1, %rdi
    nop
    nop
    
test_cond_taken3:
    mov $0xAAAA, %rsi          # This will execute
    
    # Test 30: Direct conditional branch - NOT TAKEN (JZ - zero)
    mov $0x100, %rdi           # Set RDI to non-zero
    test %rdi, %rdi            # Test RDI (ZF will NOT be set)
    jz test_cond_not_taken3    # Direct conditional branch - WILL NOT BE TAKEN (ZF is NOT set)
    
    # This code WILL execute
    mov $0xBBBB, %r8           # This will execute
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x9999, %r9
    mov $0xAAAA, %r10
    add $1, %r9
    add $1, %r10
    sub $1, %r9
    sub $1, %r10
    nop
    nop
    
test_cond_not_taken3:
    mov $0xCCCC, %r9           # This will also execute
    
    # Test 31: Indirect branch via register (JMP *%reg)
    lea test_indirect_target1(%rip), %r10
    jmp *%r10                  # Indirect branch via R10
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0xBBBB, %r11
    mov $0xCCCC, %r12
    add $1, %r11
    add $1, %r12
    sub $1, %r11
    sub $1, %r12
    nop
    nop
    
test_indirect_target1:
    mov $0xDDDD, %r11
    
    # Test 32: Indirect branch via register (JMP *%reg) - another example
    lea test_indirect_target2(%rip), %r12
    mov %r12, %r13             # Move address to R13
    jmp *%r13                  # Indirect branch via R13
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0xDDDD, %r14
    mov $0xEEEE, %r15
    add $1, %r14
    add $1, %r15
    sub $1, %r14
    sub $1, %r15
    nop
    nop
    
test_indirect_target2:
    mov $0xEEEE, %r14
    
    # Test 33: Indirect branch via memory (JMP *mem)
    lea test_indirect_target3(%rip), %r15
    mov %r15, test_var(%rip)   # Store address in memory
    jmp *test_var(%rip)        # Indirect branch via memory
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0xFFFF, %rax
    mov $0x0000, %rbx
    add $1, %rax
    add $1, %rbx
    sub $1, %rax
    sub $1, %rbx
    nop
    nop
    
test_indirect_target3:
    mov $0xFFFF, %rax
    
    # Test 34: Indirect branch via register with different condition setup
    mov $0x1234, %rbx
    lea test_indirect_target4(%rip), %rcx
    add $0, %rcx               # Ensure RCX has the address (dependency chain)
    jmp *%rcx                  # Indirect branch via RCX
    
    # Dummy instructions to create IP difference
    nop
    nop
    mov $0x1234, %rdx
    mov $0x5678, %rsi
    add $1, %rdx
    add $1, %rsi
    sub $1, %rdx
    sub $1, %rsi
    nop
    nop
    
test_indirect_target4:
    mov $0xABCD, %rdx
    
    # Return
    pop %rbp
    ret

