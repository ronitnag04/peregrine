"""
Parser class for PIN instruction trace CSV files. Converts
trace into a list of instruction objects.
"""
import csv
from collections.abc import Iterator

import numpy as np
from evantrace.x86.branch_types import Branch_Type
from evantrace.x86.instructions import Instruction
from evantrace.x86.categories import opclass_to_fu_group

class Parser:
    """
    Parses an instruction trace CSV file into a list of Instruction objects.
    """
    
    def __init__(self, filepath: str):
        """
        Initializes the parser with the path to the CSV file.
        
        Args:
            filepath (str): The path to the instructions.csv file.
        """
        self.filepath: str = filepath

    # --- Public Main Methods ---

    def iter_instructions(self) -> Iterator[Instruction]:
        """
        Lazily parses the CSV file specified in the constructor, yielding
        one Instruction at a time. This avoids materializing the full trace
        in memory at once.
        """
        with open(self.filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                branch_type = self._parse_branch_type(row['Branch Type'])
                if branch_type is None:
                    branch_taken = False
                    branch_target_addr = 0
                else:
                    branch_taken = self._parse_bool(row['Branch Taken'])
                    # TODO: We don't track branch target address in gem5 tracer yet
                    branch_target_addr = 0 # np.uint64(int(row['Branch Target Address'], 16)) 

                yield Instruction(
                    inst_ptr=np.uint64(int(row['IP'], 16)),
                    assembly=row['Assembly'],
                    category=row['Category'],
                    opcode=row['Opcode'].strip(),
                    fu_group=opclass_to_fu_group(row['Category'].strip()),
                    branch_type=branch_type,
                    branch_taken=branch_taken,
                    branch_target_addr=branch_target_addr,
                    inst_sync=self._parse_bool(row['Instruction Sync']),
                    read_regs=self._parse_register_list(row['Read Registers']),
                    write_regs=self._parse_register_list(row['Write Registers']),
                    reg_dependent_ips=self._parse_addr_list(row['Register Dependent IPs']),
                    read_addrs=self._parse_addr_list(row['Read Addresses']),
                    write_addrs=self._parse_addr_list(row['Write Addresses']),
                    mem_dependent_ips=self._parse_addr_list(row['Memory Dependent IPs'])
                )

    def parse(self) -> list[Instruction]:
        """
        Parses the CSV file specified in the constructor and returns a list
        of Instruction objects. This eagerly materializes the entire trace.
        """
        return list(self.iter_instructions())

    def count_instructions(self) -> int:
        """
        Returns the number of instructions (rows) in the trace without
        materializing Instruction objects.
        """
        with open(self.filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)

    # --- Private Helper Methods ---

    @staticmethod
    def _parse_bool(s: str) -> bool:
        """Converts 'true'/'false' string to boolean."""
        return s.lower() == 'true'

    def _parse_branch_type(self, s: str) -> Branch_Type | None:
        """Converts branch type string to Branch_Type enum."""
        if s == '':
            return None
        return Branch_Type[s.strip()]

    def _parse_register_list(self, s: str) -> list[str]:
        """
        Parses a semicolon-delimited list of register names.
        The new trace format stores registers as plain strings; we keep them as-is.
        """
        if not s or not s.strip():
            return []

        return [name.strip() for name in s.split(';') if name.strip()]

    @staticmethod
    def _parse_addr_list(s: str) -> list[np.uint64]:
        """Parses a semicolon-delimited list of addresses, e.g., '0x...(8)'."""
        addrs: list[np.uint64] = []
        if not s.strip():
            return addrs
            
        addr_strings = s.split(';')
        for addr_str in addr_strings:
            addr_str = addr_str.strip()
            hex_part = addr_str.split('(')[0]
            addrs.append(np.uint64(int(hex_part, 16)))
        return addrs
    