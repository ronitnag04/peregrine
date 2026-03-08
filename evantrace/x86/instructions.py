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
        branch_taken: bool,
        branch_target_addr: np.uint64,
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
        self.branch_taken: bool = branch_taken
        self.branch_target_addr: np.uint64 = branch_target_addr
        self.inst_sync: bool = inst_sync
        self.read_regs: list[Register] = read_regs
        self.write_regs: list[Register] = write_regs
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

        # Execution latency aligned with gem5 O3 + FUPool (see gem5_fu_latencies.py):
        # - Load-only: cache response + op latency (e.g. FloatSqrt with mem = cache + 24).
        # - Store-only: 1 cycle (MemWrite) + op latency if the inst also does work (e.g. FSTP).
        # - Load+store (e.g. atomic): cache read + op latency (store completion overlapped).
        # - Non-memory: opClass latency only.
        exec_latency = 0
        num_read_addrs = len(self.read_addrs)
        num_write_addrs = len(self.write_addrs)

        # Select opcode variant based on actual memory behavior (reads/writes)
        variants = self.opcode.value  # dict: variant -> [OpClass names]

        # Helper: classify each variant by whether it reads and/or writes memory.
        read_classes = {
            "MemRead", "FloatMemRead",
            "SimdUnitStrideLoad", "SimdUnitStrideMaskLoad",
            "SimdUnitStrideSegmentedLoad", "SimdStridedLoad",
            "SimdIndexedLoad", "SimdUnitStrideFaultOnlyFirstLoad",
            "SimdUnitStrideSegmentedFaultOnlyFirstLoad",
            "SimdWholeRegisterLoad", "SimdStrideSegmentedLoad",
        }
        write_classes = {
            "MemWrite", "FloatMemWrite",
            "SimdUnitStrideStore", "SimdUnitStrideMaskStore",
            "SimdUnitStrideSegmentedStore", "SimdStridedStore",
            "SimdIndexedStore", "SimdWholeRegisterStore",
            "SimdStrideSegmentedStore",
        }

        def variant_mem_flags(op_classes):
            has_read = any(cls in read_classes for cls in op_classes)
            has_write = any(cls in write_classes for cls in op_classes)
            return has_read, has_write

        want_read = num_read_addrs > 0
        want_write = num_write_addrs > 0

        # Find variants whose microop OpClasses match the observed mem behavior.
        matching = []
        for name, op_classes in variants.items():
            has_read, has_write = variant_mem_flags(op_classes)
            if has_read == want_read and has_write == want_write:
                matching.append(name)

        if not matching:
            # Fallback: if no exact match, prefer variants that at least
            # have the right kind of memory access.
            for name, op_classes in variants.items():
                has_read, has_write = variant_mem_flags(op_classes)
                if want_read and has_read:
                    matching.append(name)
                elif want_write and has_write:
                    matching.append(name)

        if not matching:
            # Last-resort fallback: just use the first defined variant.
            variant = next(iter(variants.keys()))
        else:
            # Prefer reg_mem, then mem_reg, then reg_reg, then any.
            preference = ("reg_mem", "mem_reg", "reg_reg")
            for preferred in preference:
                if preferred in matching:
                    variant = preferred
                    break
            else:
                variant = matching[0]
        
        try:
            op_lat = self.opcode.latency(variant)
        except Exception:
            raise ValueError(f"Error estimating {variant} latency for opcode: {self.opcode.name}")
            
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

        opcode_name = fmt_enum(self.opcode) if self.opcode is not None else "None"
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
            f"  Reads (regs):        {[fmt_enum(r) for r in self.read_regs]}\n"
            f"  Writes (regs):       {[fmt_enum(r) for r in self.write_regs]}\n"
            f"  Reg Dependent IPs:   {fmt_hex_list(self.reg_dependent_ips)}\n"
            f"  Mem Reads (addrs):   {fmt_hex_list(self.read_addrs)}\n"
            f"  Mem Writes (addrs):  {fmt_hex_list(self.write_addrs)}\n"
            f"  Mem Dependent IPs:   {fmt_hex_list(self.mem_dependent_ips)}\n"
            f"  Fetch Latency:       {self.fetch_latency}\n"
            f"  Exec Latency:        {self.exec_latency}\n"
        )
    