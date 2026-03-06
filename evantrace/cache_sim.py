from evantrace.x86.instructions import Instruction
from evantrace.caches import Cache

class CacheSim:
    """
    Implements the in-order cache simulation given
    a trace of Instructions
    """
    
    def __init__(
        self,
        trace: list[Instruction],
        icache: Cache,
        dcache: Cache,
        l2cache: Cache
    ):
        self.trace: list[Instruction] = trace
        self.icache: Cache = icache
        self.dcache: Cache = dcache
        self.l2cache: Cache = l2cache

        self.failed_opcodes = set() #[Opcode, variant]

    def run(self):
        for instruction in self.trace:
            try:
                fetch_latency, exec_latency = instruction.estimate_latency(self.icache, self.dcache)
            except Exception:
                opcode = instruction.opcode.name
                num_read_addrs = len(instruction.read_addrs)
                num_write_addrs = len(instruction.write_addrs)
                if num_read_addrs == 0 and num_write_addrs == 0:
                    variant = "reg_reg"
                elif num_read_addrs > 0 and num_write_addrs == 0:
                    variant = "reg_mem"
                elif num_read_addrs == 0 and num_write_addrs > 0:
                    variant = "mem_reg"
                self.failed_opcodes.add((opcode, variant))
                # raise ValueError(f"Error estimating latency for instruction: {instruction}")

            instruction.fetch_latency = fetch_latency
            instruction.exec_latency = exec_latency
        
        if len(self.failed_opcodes) > 0:
            print(self.failed_opcodes)
            raise ValueError("Failed to estimate latency for some opcodes")
