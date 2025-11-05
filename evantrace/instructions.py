"""
Representation of parsed instructions
"""

from enum import Enum

class BRANCH_TYPE(Enum):
    DIRECT_UNCOND = 1   # Direct unconditional branches, e.g. jmp
    DIRECT_COND = 2     # Direct conditional branches, e.g. je
    INDIRECT = 3        # Indirect branches
    
class OPCODE(Enum):
    

class Instruction:
    def __init__(
        self,
        # uarch independent
        opcode: Opcode,
        register_dependencies: list[int],
        memory_dependencies: list[int],
        program_counter: int,
        icache_line: int,
        dcache_line: int | None,
        branch_type: BRANCH_TYPE,
        # uarch dependent
        execution_latency: int,
        icache_latency: int,
        branch_mispredict_rate: float
    ):
        self.register_dependencies: list[int] = register_dependencies
        self.memory_dependencies: list[int] = memory_dependencies
        self.program_counter: int = program_counter
        self.icache_line: int = icache_line
        self.dcache_line: int | None = dcache_line
        self.branch_type: BRANCH_TYPE = branch_type
        self.execution_latency: int = execution_latency
        self.icache_latency: int = icache_latency
        self.branch_mispredict_rate: float = branch_mispredict_rate