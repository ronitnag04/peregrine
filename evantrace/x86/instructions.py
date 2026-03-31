"""
Object representation of x86 instructions.
"""
import numpy as np
from evantrace.caches import Cache
from evantrace.x86.branch_types import Branch_Type
from evantrace.x86.categories import opclass_latency

class Instruction:
    def __init__(
        self,
        # uarch independent fields derived directly from trace
        assembly: str,
        category: str,
        opcode: str,
        fu_group: str,
        inst_ptr: np.uint64,
        branch_type: Branch_Type | None,
        branch_taken: bool,
        branch_target_addr: np.uint64,
        inst_sync: bool,
        read_regs: list[str],
        write_regs: list[str],
        reg_dependent_ips: list[np.uint64],
        read_addrs: list[np.uint64],
        write_addrs: list[np.uint64],
        mem_dependent_ips: list[np.uint64]
    ):
        self.assembly: str = assembly
        self.category: str = category
        self.opcode: str = opcode
        self.fu_group: str = fu_group
        self.inst_ptr: np.uint64 = inst_ptr
        self.branch_type: Branch_Type | None = branch_type
        self.branch_taken: bool = branch_taken
        self.branch_target_addr: np.uint64 = branch_target_addr
        self.inst_sync: bool = inst_sync
        self.read_regs: list[str] = read_regs
        self.write_regs: list[str] = write_regs
        self.reg_dependent_ips: list[np.uint64] = reg_dependent_ips
        self.read_addrs: list[np.uint64] = read_addrs
        self.write_addrs: list[np.uint64] = write_addrs
        self.mem_dependent_ips: list[np.uint64] = mem_dependent_ips
        
        # default values, filled in by in-order simulation
        self.fetch_latency: int = 0
        self.exec_latency: int = 0
    
    def is_load(self):
        return self.read_addrs != []
    
    def estimate_latency(self, icache: Cache, dcache: Cache):
        # Fetch latency: I-cache access (matches gem5 instruction fetch path).
        fetch_latency = icache.read(self.inst_ptr)

        # Execution latency:
        # - Base op latency from the Category string (cycles in _CATEGORY_LATENCIES, else 1).
        # - Plus data cache latency for the slowest load (if any).
        # - Plus 1 cycle for any store.
        exec_latency = 0
        num_read_addrs = len(self.read_addrs)
        num_write_addrs = len(self.write_addrs)

        op_lat = opclass_latency(self.category)
        exec_latency = op_lat
        if num_read_addrs > 0:
            exec_latency += max(dcache.read(addr) for addr in self.read_addrs)
        if num_write_addrs > 0:
            exec_latency += 1 

        return fetch_latency, exec_latency

    
    def __str__(self):
        """Returns a neat, human-readable string representation."""
        def fmt_hex(x) -> str:
            return hex(int(x))

        def fmt_enum(x) -> str:
            return getattr(x, "name", str(x))

        def fmt_hex_list(xs) -> list[str]:
            return [fmt_hex(x) for x in xs]

        opcode_name = self.opcode if self.opcode is not None else "None"
        branch_type = fmt_enum(self.branch_type) if self.branch_type is not None else "None"

        return (
            f"Instruction @ {fmt_hex(self.inst_ptr)}\n"
            f"  Assembly:            {self.assembly}\n"
            f"  Category:            {self.category}\n"
            f"  Opcode:              {opcode_name}\n"
            f"  Inst Sync:           {self.inst_sync}\n"
            f"  Branch Type:         {branch_type}\n"
            f"  Branch Taken:        {self.branch_taken}\n"
            f"  Branch Target Addr:  {fmt_hex(self.branch_target_addr)}\n"
            f"  Reads (regs):        {self.read_regs}\n"
            f"  Writes (regs):       {self.write_regs}\n"
            f"  Reg Dependent IPs:   {fmt_hex_list(self.reg_dependent_ips)}\n"
            f"  Mem Reads (addrs):   {fmt_hex_list(self.read_addrs)}\n"
            f"  Mem Writes (addrs):  {fmt_hex_list(self.write_addrs)}\n"
            f"  Mem Dependent IPs:   {fmt_hex_list(self.mem_dependent_ips)}\n"
            f"  Fetch Latency:       {self.fetch_latency}\n"
            f"  Exec Latency:        {self.exec_latency}\n"
        )
    