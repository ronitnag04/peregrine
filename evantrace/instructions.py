"""
Object representation of x86 instructions.
"""
import numpy as np
from enum import Enum
from evantrace.caches import Cache

class Branch_Type(Enum):
    direct_unconditional = 1   # Direct unconditional branches, e.g. jmp
    direct_conditional = 2     # Direct conditional branches, e.g. je
    indirect = 3               # Indirect branches

"""
Enum mapping instruction opcodes to a map of latencies for 
that opcode for every relevant combination of operand
types, e.g. register to register, register to memory.
"""
class Opcode(Enum):
    # Arithmetic
    ADD = { 'reg_reg': 1, 'reg_mem': 1, 'mem_reg': 6 }
    SUB = { 'reg_reg': 1, 'reg_mem': 1, 'mem_reg': 6 }
    AND = { 'reg_reg': 1, 'reg_mem': 1, 'mem_reg': 6 }
    OR  = { 'reg_reg': 1, 'reg_mem': 1, 'mem_reg': 6 }
    XOR = { 'reg_reg': 1, 'reg_mem': 1, 'mem_reg': 6 }
    SHL = { 'reg_reg': 1, 'reg_mem': 1, 'mem_reg': 6 }
    TEST = {'reg_reg': 1, 'mem_reg': 1 }
    # Data Transfer
    MOV = { 'reg_reg': 1, 'reg_mem': 2, 'mem_reg': 3}
    PUSH = { 'reg_mem': 2 }
    LEA = { 'reg_reg': 1, 'reg_mem': 1 }
    # Control Transfer
    CALL_NEAR = { 'reg_mem': 2 }
    # Instruction Synchronization / System
    RDTSC = { 'reg_reg': 0 }
    # Not sure yet
    NOP = { 'reg_reg': 1 }
    
    
    # TODO: lots more opcodes and latency mappings
    
    """Helper methods for accessing latency values"""
    def reg_reg_latency(self):
        try:
            return self.value['reg_reg']
        except KeyError:
            raise ValueError(f"Opcode {str(self)} does not have a reg_reg latency.")
            
    def reg_mem_latency(self):
        try:
            return self.value['reg_mem']
        except KeyError:
            raise ValueError(f"Opcode {str(self)} does not have a reg_mem latency.")
            
    def mem_reg_latency(self):
        try:
            return self.value['mem_reg']
        except KeyError:
            raise ValueError(f"Opcode {str(self)} does not have a mem_reg latency.")
    
class Register(Enum):
    # 64 bit general purpose
    rax = 0
    rbx = 1
    rcx = 2
    rdx = 3
    rsi = 4
    rdi = 5
    rsp = 6
    rbp = 7
    r8 = 8
    r9 = 9
    r10 = 10
    r11 = 11
    r12 = 12
    r13 = 13
    r14 = 14
    r15 = 15
    # 32 bit general purpose
    eax = 16
    ebx = 17
    ecx = 18
    edx = 19
    esi = 20
    edi = 21
    esp = 22
    ebp = 23
    r8d = 24
    r9d = 25
    r10d = 26
    r11d = 27
    r12d = 28
    r13d = 29
    r14d = 30
    r15d = 31
    # Pointer
    rip = 32    # 64 bit
    eip = 33    # 32 bit
    ip = 34     # 16 bit
    # Flags
    rflags = 35
    # TODO: different size registers, check with Ronit
    # to see how overlap dependencies will look
    

class Instruction:
    def __init__(
        self,
        # uarch independent fields derived directly from trace
        assembly: str,
        category: str,
        opcode: Opcode,
        inst_ptr: np.uint64,
        branch_type: Branch_Type | None,
        inst_sync: bool,
        read_regs: list[Register],
        write_regs: list[Register],
        reg_dependent_ips: list[np.uint64],
        read_addrs: list[np.uint64],
        write_addrs: list[np.uint64],
        mem_dependent_ips: list[np.uint64]
    ):
        self.assembly: str = assembly
        self.category: str = category
        self.opcode: Opcode = opcode
        self.inst_ptr: np.uint64 = inst_ptr
        self.branch_type: Branch_Type | None = branch_type
        self.inst_sync: bool = inst_sync
        self.read_regs: list[Register] = read_regs
        self.write_regs: list[Register] = write_regs
        self.reg_dependent_ips: list[np.uint64] = reg_dependent_ips
        self.read_addrs: list[np.uint64] = read_addrs
        self.write_addrs: list[np.uint64] = write_addrs
        self.mem_dependent_ips: list[np.uint64] = mem_dependent_ips
        
        self.exec_latency: int = -1 # default value, filled in by in-order simulation
    
    def is_load(self):
        return self.read_addrs != []
    
    def get_latency(self, icache: Cache, dcache: Cache):
        # first add instruction fetch latency based on icache simulation
        latency = icache.read(self.inst_ptr)
        
        # next add op latency (does not include memory latency)
        if len(self.write_addrs) > 0:
            latency += self.opcode.mem_reg_latency()
        elif len(self.read_addrs) > 0:
            latency += self.opcode.reg_mem_latency()
        else:
            latency += self.opcode.reg_reg_latency()
        
        # then add memory latency based on dcache simulation
        if len(self.read_addrs) > 0:
            # if read addresses is not empty, there must be a load
            latency += dcache.read(self.read_addrs[0]) 
        
        if len(self.write_addrs) > 0:
            # must be a mem-to-reg (load) operation
            # invoke cache model
            latency += dcache.write(self.write_addrs[0])
            
            
        return latency
    