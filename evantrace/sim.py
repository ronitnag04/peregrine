from evantrace.x86.instructions import Instruction
from evantrace.caches import Cache
from evantrace.branch_predictor import BranchPredictor

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
        l2cache: Cache,
        branch_predictor: BranchPredictor
    ):
        self.trace: list[Instruction] = trace
        self.icache: Cache = icache
        self.dcache: Cache = dcache
        self.l2cache: Cache = l2cache
        self.branch_predictor: BranchPredictor = branch_predictor

    def run(self):
        for instruction in self.trace:
            try:
                fetch_latency, exec_latency = instruction.estimate_latency(self.icache, self.dcache)
            except Exception:
                print(instruction)
                raise
                
            instruction.fetch_latency = fetch_latency
            instruction.exec_latency = exec_latency

            predicted_taken = self.branch_predictor.predict(instruction.inst_ptr, instruction.branch_type)  
            self.branch_predictor.update(instruction.inst_ptr, instruction.branch_type, predicted_taken, instruction.branch_taken)