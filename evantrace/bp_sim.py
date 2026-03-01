from evantrace.x86.instructions import Instruction
from evantrace.branch_predictor import BranchPredictor

class BPSim:
    """
    Implements the branch predictor simulation given
    a trace of Instructions
    """
    
    def __init__(
        self,
        trace: list[Instruction],
        branch_predictor: BranchPredictor
    ):
        self.trace: list[Instruction] = trace
        self.branch_predictor: BranchPredictor = branch_predictor

    def run(self):
        for instruction in self.trace:
            predicted_taken = self.branch_predictor.predict(instruction.inst_ptr, instruction.branch_type)  
            self.branch_predictor.update(instruction.inst_ptr, instruction.branch_type, predicted_taken, instruction.branch_taken)