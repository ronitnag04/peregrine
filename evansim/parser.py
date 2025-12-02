"""
Parser class for PIN instruction trace CSV files. Converts
trace into a list of instruction objects.
"""
import csv
import numpy as np
from evantrace.x86.opcodes import Opcode
from evantrace.x86.registers import Register
from evantrace.x86.branch_types import Branch_Type
from evantrace.x86.instructions import Instruction

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

    # --- Public Main Method ---
    
    def parse(self) -> list[Instruction]:
        """
        Parses the CSV file specified in the constructor.
        
        Returns:
            List[Instruction]: A list of parsed Instruction objects.
        """
        instructions: list[Instruction] = []
        
        try:
            with open(self.filepath, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    try:
                        opcode = self._parse_opcode(row['Opcode'])
                        if opcode is None:
                            print(f"Skipping row {i+2}: Missing or unknown opcode '{row['Opcode']}'.")
                            continue

                        inst = Instruction(
                            inst_ptr=np.uint64(int(row['IP'], 16)),
                            assembly=row['Assembly'],
                            category=row['Category'],
                            opcode=opcode,
                            branch_type=self._parse_branch_type(row['Branch Type']),
                            inst_sync=self._parse_bool(row['Instruction Sync']),
                            read_regs=self._parse_register_list(row['Read Registers']),
                            write_regs=self._parse_register_list(row['Write Registers']),
                            reg_dependent_ips=self._parse_addr_list(row['Register Dependent IPs']),
                            read_addrs=self._parse_addr_list(row['Read Addresses']),
                            write_addrs=self._parse_addr_list(row['Write Addresses']),
                            mem_dependent_ips=self._parse_addr_list(row['Memory Dependent IPs'])
                        )
                        instructions.append(inst)
                    
                    except Exception as e:
                        raise ValueError(f"Error processing row {i+2}: {e}\nRow data: {row}")
        
        except FileNotFoundError:
            print(f"Error: File not found at '{self.filepath}'")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

        return instructions

    # --- Private Helper Methods ---

    @staticmethod
    def _parse_bool(s: str) -> bool:
        """Converts 'true'/'false' string to boolean."""
        return s.lower() == 'true'

    def _parse_branch_type(self, s: str) -> Branch_Type | None:
        """Converts branch type string to Branch_Type enum."""
        if s == '':
            return None
        try:
            return Branch_Type[s.strip()]
        except KeyError:
            raise ValueError(f"Not a valid branch type: {s}")

    def _parse_opcode(self, s: str) -> Opcode | None:
        """Converts opcode string to Opcode enum."""
        try:
            return Opcode[s.strip()]
        except KeyError:
            raise ValueError(f"Not a valid opcode: {s}")

    def _parse_register_list(self, s: str) -> list[Register]:
        """Parses a semicolon-delimited list of register names."""
        registers: list[Register] = []
        if not s.strip():
            return registers
        
        reg_names = s.split(';')
        for name in reg_names:
            name = name.strip()
            if not name:
                continue
            try:
                reg = Register[name]
            except KeyError:
                raise ValueError(f"Not a valid register: {name}")
            registers.append(reg)
    
        return registers

    @staticmethod
    def _parse_addr_list(s: str) -> list[np.uint64]:
        """Parses a semicolon-delimited list of addresses, e.g., '0x...(8)'."""
        addrs: list[np.uint64] = []
        if not s.strip():
            return addrs
            
        addr_strings = s.split(';')
        for addr_str in addr_strings:
            addr_str = addr_str.strip()
            if not addr_str:
                continue
            
            hex_part = addr_str.split('(')[0]
            try:
                addrs.append(np.uint64(int(hex_part, 16)))
            except ValueError:
                raise ValueError(f"Could not parse address '{hex_part}'.")
        return addrs
    