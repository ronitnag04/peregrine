from collections.abc import Iterable, Iterator

from evantrace.x86.instructions import Instruction
from evantrace.caches import Cache

class CacheSim:
    """
    Implements the in-order cache simulation given
    a trace of Instructions
    """
    
    def __init__(
        self,
        trace: Iterable[Instruction],
        icache: Cache,
        dcache: Cache,
        l2cache: Cache,
    ):
        self.trace: Iterable[Instruction] = trace
        self.icache: Cache = icache
        self.dcache: Cache = dcache
        self.l2cache: Cache = l2cache

    def run(self) -> Iterator[tuple[int, int]]:
        """
        Runs the cache simulation over the trace, yielding a tuple
        (fetch_latency, exec_latency) for each instruction in order.
        """
        for instruction in self.trace:
            fetch_latency, exec_latency = instruction.estimate_latency(self.icache, self.dcache)
            yield fetch_latency, exec_latency
