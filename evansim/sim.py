from evantrace.x86.instructions import Instruction
from evantrace.caches import Cache

class Sim:
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
        
    def run(self):
        for instruction in self.trace:
            try:
                fetch_latency, exec_latency = instruction.estimate_latency(self.icache, self.dcache)
            except Exception:
                print(instruction)
                raise
                
            instruction.fetch_latency = fetch_latency
            instruction.exec_latency = exec_latency