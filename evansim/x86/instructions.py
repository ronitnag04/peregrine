"""
Object representation of x86 instructions.
"""
import numpy as np
from evantrace.caches import Cache
from evantrace.x86.opcodes import Opcode
from evantrace.x86.registers import Register
from evantrace.x86.branch_types import Branch_Type
    
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
        
        self.latency: int = 0 # default value, filled in by in-order simulation
    
    def is_load(self):
        return self.read_addrs != []
    
    def estimate_latency(self, icache: Cache, dcache: Cache):
        # first add instruction fetch latency based on icache simulation
        latency = icache.read(self.inst_ptr)
        
        
        # next add op latency (does not include memory latency)
        if len(self.read_addrs) > 0:
            latency += self.opcode.mem_reg_latency()
        elif len(self.write_addrs) > 0:
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
    
    def __repr__(self):
            """Helper for printing the object nicely."""
            return (
                f"Instruction(IP: {hex(self.inst_ptr)}, "
                f"Opcode: {self.opcode.name}, "
                f"Assembly: \"{self.assembly}\")"
            )
    
    def __str__(self):
        """Returns a neat, human-readable string representation."""
        return (
            f"Instruction @ {hex(self.inst_ptr)}\n"
            f"  Assembly:   {self.assembly}\n"
            f"  Category:   {self.category}\n"
            f"  Opcode:     {self.opcode.name if self.opcode else 'None'}\n"
            f"  Reads:      {[r.name for r in self.read_regs]}\n"
            f"  Writes:     {[r.name for r in self.write_regs]}\n"
            f"  Mem Reads:  {[hex(a) for a in self.read_addrs]}\n"
            f"  Mem Writes: {[hex(a) for a in self.write_addrs]}\n"
            f"  Latency:    {self.latency}"
        )
    