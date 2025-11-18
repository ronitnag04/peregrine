from enum import Enum

class Branch_Type(Enum):
    """
    Enum of the three types of branches present in
    the x86 instruction set.
    """
    direct_unconditional = 1   # Direct unconditional branches, e.g. jmp
    direct_conditional = 2     # Direct conditional branches, e.g. je
    indirect = 3               # Indirect branches