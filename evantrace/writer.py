import csv
import numpy as np
from typing import List, Any
from evantrace.x86.instructions import Instruction

class Writer:
    """
    Writes a list of Instruction objects to a CSV file, including a computed
    Latency column.
    """

    def __init__(self, filepath: str):
        """
        Initializes the writer with the output file path.

        Args:
            filepath (str): The path where the CSV will be written.
        """
        self.filepath = filepath
        self.fieldnames = [
            "IP",
            "Assembly",
            "Category",
            "Opcode",
            "Branch Type",
            "Branch Taken",
            "Branch Target Address",
            "Instruction Sync",
            "Read Registers",
            "Write Registers",
            "Register Dependent IPs",
            "Read Addresses",
            "Write Addresses",
            "Memory Dependent IPs",
            # New columns
            "Fetch Latency",
            "Execution Latency"
        ]

    def write(self, instructions: List[Instruction]) -> None:
        """
        Writes the list of instructions to the CSV file.

        Args:
            instructions (List[Instruction]): The data to write.
        """
        try:
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

                for inst in instructions:
                    row = {
                        "IP": f"{inst.inst_ptr:#x}", # Formats as 0x...
                        "Assembly": inst.assembly,
                        "Category": inst.category,
                        "Opcode": inst.opcode or "",
                        "Branch Type": inst.branch_type.name if inst.branch_type else "",
                        "Branch Taken": str(inst.branch_taken).lower() if inst.branch_type else "",
                        "Branch Target Address": f"{inst.branch_target_addr:#x}" if inst.branch_type else "",
                        "Instruction Sync": str(inst.inst_sync).lower(),
                        "Read Registers": self._format_reg_list(inst.read_regs),
                        "Write Registers": self._format_reg_list(inst.write_regs),
                        "Register Dependent IPs": self._format_hex_list(inst.reg_dependent_ips),
                        "Read Addresses": self._format_hex_list(inst.read_addrs),
                        "Write Addresses": self._format_hex_list(inst.write_addrs),
                        "Memory Dependent IPs": self._format_hex_list(inst.mem_dependent_ips),
                        "Fetch Latency": inst.fetch_latency,
                        "Execution Latency": inst.exec_latency
                    }
                    writer.writerow(row)
            
            print(f"Successfully wrote {len(instructions)} instructions to '{self.filepath}'")

        except IOError as e:
            print(f"Error writing to file '{self.filepath}': {e}")

    def _format_reg_list(self, items: List[Any]) -> str:
        """Joins register names with semicolons (items are plain strings)."""
        if not items:
            return ""
        return ";".join(str(item) for item in items)

    def _format_hex_list(self, items: List[np.uint64]) -> str:
        """Joins hex numbers with semicolons."""
        if not items:
            return ""
        # Note: This outputs pure hex (0x123). If strict '0x123(8)' format 
        # is required, we would need the size info which isn't in the current model.
        return ";".join([f"{item:#x}" for item in items])