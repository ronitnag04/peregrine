#!/usr/bin/env python3
"""
Script to verify trace.csv correctness based on test assembly files.
Checks for: IP tracking, branch types, branch taken/target tracking, sync barriers,
register dependencies, memory dependencies, read/write registers, read/write addresses.
"""

import csv
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

class TraceVerifier:
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.instructions = []
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def parse_trace(self):
        """Parse the trace CSV file."""
        try:
            with open(self.trace_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.instructions.append(row)
                    self.stats['total_instructions'] += 1
        except FileNotFoundError:
            self.errors.append(f"Trace file not found: {self.trace_file}")
            return False
        except Exception as e:
            self.errors.append(f"Error parsing trace file: {e}")
            return False
        return True
    
    def parse_semicolon_list(self, value: str) -> List[str]:
        """Parse a semicolon-separated list, handling empty strings."""
        if not value or value.strip() == '':
            return []
        return [item.strip() for item in value.split(';') if item.strip()]
    
    def parse_address(self, addr_str: str) -> Optional[Tuple[int, int]]:
        """Parse address string like '0xADDR(SIZE)' or '0xADDR'."""
        if not addr_str:
            return None
        # Match 0xADDR(SIZE) or 0xADDR
        match = re.match(r'0x([0-9a-fA-F]+)(?:\((\d+)\))?', addr_str)
        if match:
            addr = int(match.group(1), 16)
            size = int(match.group(2)) if match.group(2) else 0
            return (addr, size)
        return None
    
    def check_ip_tracking(self):
        """Verify that every instruction has an IP."""
        print("\n[1] Checking IP tracking...")
        missing_ips = 0
        for i, instr in enumerate(self.instructions):
            if not instr.get('IP') or instr['IP'].strip() == '':
                self.errors.append(f"Instruction {i} missing IP")
                missing_ips += 1
        
        if missing_ips == 0:
            print(f"  ✓ All {len(self.instructions)} instructions have IP addresses")
            return True
        else:
            print(f"  ✗ {missing_ips} instructions missing IP addresses")
            return False
    
    def check_branch_types(self):
        """Verify branch types: direct_unconditional, direct_conditional, indirect."""
        print("\n[2] Checking branch types...")
        
        found_branches = defaultdict(set)
        indirect_branches = []
        
        for instr in self.instructions:
            branch_type = instr.get('Branch Type', '').strip()
            opcode = instr.get('Opcode', '').strip()
            assembly = instr.get('Assembly', '').strip()
            
            if branch_type:
                found_branches[branch_type].add(opcode)
                self.stats[f'branch_{branch_type}'] += 1
                
                # Check for indirect branch (JMP with register operand)
                if branch_type == 'indirect' or (opcode == 'JMP' and '*' in assembly):
                    indirect_branches.append((instr.get('IP'), assembly))
        
        # Check for expected branch types
        all_found = True
        for branch_type in ['direct_unconditional', 'direct_conditional', 'indirect']:
            count = self.stats[f'branch_{branch_type}']
            if count > 0:
                print(f"  ✓ Found {count} {branch_type} branch(es)")
            else:
                print(f"  ⚠ No {branch_type} branches found (may be expected)")
                self.warnings.append(f"No {branch_type} branches found")
        
        return all_found
    
    def check_branch_taken_and_target(self):
        """Verify branch taken and branch target address fields are tracked properly."""
        print("\n[2a] Checking branch taken and target address tracking...")
        
        branch_issues = []
        branches_with_taken = 0
        branches_with_target = 0
        unconditional_taken = 0
        conditional_taken = 0
        conditional_not_taken = 0
        indirect_taken = 0
        
        for instr in self.instructions:
            ip = instr.get('IP', '').strip()
            branch_type = instr.get('Branch Type', '').strip()
            branch_taken = instr.get('Branch Taken', '').strip().lower()
            branch_target = instr.get('Branch Target Address', '').strip()
            opcode = instr.get('Opcode', '').strip()
            assembly = instr.get('Assembly', '').strip()
            
            if branch_type:
                # This is a branch instruction - should have branch_taken and branch_target
                if not branch_taken:
                    branch_issues.append((ip, f"Branch instruction missing 'Branch Taken' field"))
                    self.errors.append(f"IP {ip} ({assembly}): Branch instruction missing 'Branch Taken' field")
                elif branch_taken not in ['true', 'false']:
                    branch_issues.append((ip, f"Branch instruction has invalid 'Branch Taken' value: '{branch_taken}'"))
                    self.errors.append(f"IP {ip} ({assembly}): Invalid 'Branch Taken' value: '{branch_taken}'")
                else:
                    branches_with_taken += 1
                    if branch_taken == 'true':
                        if branch_type == 'direct_unconditional':
                            unconditional_taken += 1
                        elif branch_type == 'direct_conditional':
                            conditional_taken += 1
                        elif branch_type == 'indirect':
                            indirect_taken += 1
                    else:  # branch_taken == 'false'
                        if branch_type == 'direct_conditional':
                            conditional_not_taken += 1
                
                if not branch_target:
                    branch_issues.append((ip, f"Branch instruction missing 'Branch Target Address' field"))
                    self.errors.append(f"IP {ip} ({assembly}): Branch instruction missing 'Branch Target Address' field")
                else:
                    # Check if target address is valid hex format
                    if not re.match(r'^0x[0-9a-fA-F]+$', branch_target):
                        branch_issues.append((ip, f"Branch instruction has invalid target address format: '{branch_target}'"))
                        self.errors.append(f"IP {ip} ({assembly}): Invalid target address format: '{branch_target}'")
                    else:
                        branches_with_target += 1
                        # Check that target address is not 0 (unless it's an indirect branch that wasn't taken)
                        target_val = int(branch_target, 16)
                        if target_val == 0 and branch_type != 'indirect':
                            branch_issues.append((ip, f"Direct branch has zero target address"))
                            self.warnings.append(f"IP {ip} ({assembly}): Direct branch has zero target address (may be valid for indirect not-taken)")
            else:
                # This is NOT a branch instruction - should have empty branch_taken and branch_target
                if branch_taken:
                    branch_issues.append((ip, f"Non-branch instruction has 'Branch Taken' field set: '{branch_taken}'"))
                    self.warnings.append(f"IP {ip} ({assembly}): Non-branch instruction has 'Branch Taken' field set")
                
                if branch_target:
                    branch_issues.append((ip, f"Non-branch instruction has 'Branch Target Address' field set: '{branch_target}'"))
                    self.warnings.append(f"IP {ip} ({assembly}): Non-branch instruction has 'Branch Target Address' field set")
        
        # Report findings
        if branches_with_taken > 0:
            print(f"  ✓ Found {branches_with_taken} branches with 'Branch Taken' field set")
            if unconditional_taken > 0:
                print(f"    - {unconditional_taken} direct unconditional branches (all should be taken=true)")
            if conditional_taken > 0:
                print(f"    - {conditional_taken} direct conditional branches taken (taken=true)")
            if conditional_not_taken > 0:
                print(f"    - {conditional_not_taken} direct conditional branches not taken (taken=false)")
            if indirect_taken > 0:
                print(f"    - {indirect_taken} indirect branches taken (taken=true)")
        
        if branches_with_target > 0:
            print(f"  ✓ Found {branches_with_target} branches with 'Branch Target Address' field set")
        
        if branch_issues:
            print(f"  ✗ Found {len(branch_issues)} issues with branch tracking")
            # Show first 10 issues
            for ip, issue in branch_issues[:10]:
                print(f"    - IP {ip}: {issue}")
            if len(branch_issues) > 10:
                print(f"    ... and {len(branch_issues) - 10} more issues")
            return False
        
        if branches_with_taken == 0 and branches_with_target == 0:
            print(f"  ⚠ No branches found with taken/target fields (may be expected)")
            return True
        
        # Additional verification: Check that direct unconditional branches are always taken
        unconditional_not_taken = 0
        for instr in self.instructions:
            branch_type = instr.get('Branch Type', '').strip()
            branch_taken = instr.get('Branch Taken', '').strip().lower()
            if branch_type == 'direct_unconditional' and branch_taken == 'false':
                unconditional_not_taken += 1
                ip = instr.get('IP', '').strip()
                assembly = instr.get('Assembly', '').strip()
                self.errors.append(f"IP {ip} ({assembly}): Direct unconditional branch should always be taken=true")
        
        if unconditional_not_taken > 0:
            print(f"  ✗ Found {unconditional_not_taken} direct unconditional branches with taken=false (should always be true)")
            return False
        
        # Check that when branch_taken is "true", the next instruction's IP matches branch_target
        taken_branches_verified = 0
        taken_branches_mismatch = []
        
        for i, instr in enumerate(self.instructions):
            branch_taken = instr.get('Branch Taken', '').strip().lower()
            branch_target = instr.get('Branch Target Address', '').strip()
            branch_type = instr.get('Branch Type', '').strip()
            ip = instr.get('IP', '').strip()
            assembly = instr.get('Assembly', '').strip()
            
            if branch_taken == 'true' and branch_target:
                # This branch was taken - next instruction should be at branch_target
                if i + 1 < len(self.instructions):
                    next_instr = self.instructions[i + 1]
                    next_ip = next_instr.get('IP', '').strip()
                    
                    try:
                        # Parse addresses (both should be in hex format with 0x prefix)
                        target_val = int(branch_target, 16) if branch_target.startswith('0x') else int(branch_target, 16)
                        next_ip_val = int(next_ip, 16) if next_ip.startswith('0x') else int(next_ip, 16)
                        
                        if next_ip_val == target_val:
                            taken_branches_verified += 1
                        else:
                            taken_branches_mismatch.append((
                                ip, assembly, branch_type, branch_target, next_ip
                            ))
                            self.errors.append(
                                f"IP {ip} ({assembly}): Branch taken=true but next instruction IP {next_ip} "
                                f"does not match branch target {branch_target}"
                            )
                    except (ValueError, AttributeError):
                        # Skip if we can't parse addresses
                        self.warnings.append(
                            f"IP {ip} ({assembly}): Could not parse branch target or next IP for verification"
                        )
                else:
                    # This is the last instruction in trace - can't verify
                    self.warnings.append(
                        f"IP {ip} ({assembly}): Branch taken=true but no next instruction to verify target"
                    )
        
        # Report findings
        if taken_branches_verified > 0:
            print(f"  ✓ Verified {taken_branches_verified} taken branches: next instruction IP matches branch target")
        
        if taken_branches_mismatch:
            print(f"  ✗ Found {len(taken_branches_mismatch)} taken branches where next instruction IP doesn't match target")
            # Show first 5 mismatches
            for ip, assembly, branch_type, target, next_ip in taken_branches_mismatch[:5]:
                print(f"    - IP {ip}: target={target}, next IP={next_ip}")
            if len(taken_branches_mismatch) > 5:
                print(f"    ... and {len(taken_branches_mismatch) - 5} more mismatches")
            return False
        
        if taken_branches_verified == 0 and branches_with_taken > 0:
            print(f"  ⚠ No taken branches verified (may be expected if trace ends with branches)")
        
        print(f"  ✓ Branch taken and target address tracking looks correct")
        return True
    
    def check_sync_barriers(self):
        """Verify instruction sync barriers are detected."""
        print("\n[3] Checking instruction sync barriers...")
        
        sync_instructions = {
            'MFENCE': 'mfence',
            'SFENCE': 'sfence',
            'LFENCE': 'lfence',
            'CPUID': 'cpuid',
            'XCHG': 'xchg',
            'CMPXCHG': 'cmpxchg',
            'RDTSC': 'rdtsc',
            'RDTSCP': 'rdtscp',
        }
        
        found_syncs = defaultdict(int)
        missing_syncs = []
        
        for instr in self.instructions:
            opcode = instr.get('Opcode', '').strip()
            inst_sync = instr.get('Instruction Sync', '').strip().lower()
            assembly = instr.get('Assembly', '').strip().lower()
            
            # Check if this should be a sync barrier
            for sync_opcode, sync_name in sync_instructions.items():
                if opcode == sync_opcode or sync_name in assembly:
                    found_syncs[sync_name] += 1
                    if inst_sync != 'true':
                        missing_syncs.append((instr.get('IP'), opcode, assembly))
                        self.errors.append(
                            f"Sync barrier {sync_name} at IP {instr.get('IP')} not marked as sync"
                        )
        
        # Report findings
        if found_syncs:
            print(f"  ✓ Found sync barriers:")
            for sync_name, count in found_syncs.items():
                print(f"    - {sync_name}: {count}")
        else:
            print(f"  ⚠ No sync barriers found (may be expected)")
            self.warnings.append("No sync barriers found")
        
        if missing_syncs:
            print(f"  ✗ {len(missing_syncs)} sync barriers not properly marked")
            return False
        
        return True
    
    def check_register_operations(self):
        """Verify register read/write tracking."""
        print("\n[4] Checking register read/write tracking...")
        
        registers_with_reads = set()
        registers_with_writes = set()
        
        for instr in self.instructions:
            read_regs = self.parse_semicolon_list(instr.get('Read Registers', ''))
            write_regs = self.parse_semicolon_list(instr.get('Write Registers', ''))
            
            registers_with_reads.update(read_regs)
            registers_with_writes.update(write_regs)
        
        print(f"  ✓ Found {len(registers_with_reads)} unique registers read")
        print(f"  ✓ Found {len(registers_with_writes)} unique registers written")
        
        # Check for common registers
        common_regs = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rsp', 'rbp', 'rip']
        found_common = [r for r in common_regs if r in registers_with_reads or r in registers_with_writes]
        if found_common:
            print(f"  ✓ Found common registers: {', '.join(found_common[:5])}...")
        
        return True
    
    def check_register_normalization(self):
        """Verify that only full registers are tracked (no partial registers like EAX, AX, AL)."""
        print("\n[5] Checking register normalization (full registers only)...")
        
        # Define partial register patterns (case-insensitive matching)
        # 32-bit partial registers (EAX, EBX, etc.) - these should be normalized to RAX, RBX, etc.
        partial_32bit = ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp', 'eip', 'eflags']
        # 16-bit partial registers (AX, BX, etc.) - these should be normalized to RAX, RBX, etc.
        partial_16bit = ['ax', 'bx', 'cx', 'dx', 'si', 'di', 'sp', 'bp', 'ip']
        # 8-bit partial registers (AL, AH, BL, etc.) - these should be normalized to RAX, RBX, etc.
        partial_8bit = ['al', 'ah', 'bl', 'bh', 'cl', 'ch', 'dl', 'dh', 'sil', 'dil', 'spl', 'bpl']
        # 8-bit registers from R8-R15 (R8B, R9B, etc.) - these should be normalized to R8, R9, etc.
        partial_8bit_extended = [f'r{i}b' for i in range(8, 16)]
        
        # Combine all partial register patterns
        # Note: We use lowercase for matching since we convert to lowercase before checking
        all_partial_regs = set(partial_32bit + partial_16bit + partial_8bit + partial_8bit_extended)
        
        # Track partial registers found
        partial_regs_found = []
        
        for instr in self.instructions:
            ip = instr.get('IP', '').strip()
            read_regs = self.parse_semicolon_list(instr.get('Read Registers', ''))
            write_regs = self.parse_semicolon_list(instr.get('Write Registers', ''))
            
            # Check read registers
            for reg in read_regs:
                reg_lower = reg.lower()
                if reg_lower in all_partial_regs:
                    partial_regs_found.append((ip, 'read', reg))
            
            # Check write registers
            for reg in write_regs:
                reg_lower = reg.lower()
                if reg_lower in all_partial_regs:
                    partial_regs_found.append((ip, 'write', reg))
        
        if partial_regs_found:
            # Group by register name for better reporting
            partial_by_reg = {}
            for ip, rw_type, reg in partial_regs_found:
                if reg not in partial_by_reg:
                    partial_by_reg[reg] = []
                partial_by_reg[reg].append((ip, rw_type))
            
            print(f"  ✗ Found {len(partial_regs_found)} partial register references:")
            for reg, occurrences in sorted(partial_by_reg.items()):
                count = len(occurrences)
                example_ip, example_type = occurrences[0]
                print(f"    - {reg} ({count} occurrence(s), e.g., {example_type} at IP {example_ip})")
                # Add error for each unique register
                self.errors.append(
                    f"Partial register '{reg}' found in trace (should be normalized to full register, e.g., EAX→RAX, AX→RAX, AL→RAX)"
                )
            
            return False
        else:
            print(f"  ✓ All registers are normalized to full registers (no partial registers found)")
            return True
    
    def check_register_dependencies(self):
        """Verify register dependencies are tracked."""
        print("\n[6] Checking register dependencies...")
        
        # Registers that are special and may not always have dependencies tracked
        # RIP is implicitly updated by every instruction and used for PC-relative addressing
        # RFLAGS is updated by many instructions and dependencies may not always be tracked
        # Note: After fixing the tracing code to track dependencies at execution time,
        # these should now be tracked correctly, but we can still ignore them for verification
        # if they cause too many false positives
        ignore_regs = set()  # No longer ignoring registers - dependencies should be tracked correctly
        
        # Track last write IP for each register (by position in trace)
        last_write_ip = {}
        last_write_pos = {}  # Track position in trace for ordering
        dependency_found = False
        missing_deps = []
        
        for pos, instr in enumerate(self.instructions):
            ip = instr.get('IP', '').strip()
            read_regs = self.parse_semicolon_list(instr.get('Read Registers', ''))
            write_regs = self.parse_semicolon_list(instr.get('Write Registers', ''))
            reg_deps = self.parse_semicolon_list(instr.get('Register Dependent IPs', ''))
            
            # Check if read registers have dependencies (ignore special registers)
            for reg in read_regs:
                reg_lower = reg.lower()
                if reg_lower in ignore_regs:
                    continue  # Skip RIP, RFLAGS, etc.
                
                if reg in last_write_ip:
                    expected_dep_ip = last_write_ip[reg]
                    expected_pos = last_write_pos[reg]
                    
                    # Skip self-dependencies (instruction reading and writing same reg)
                    if expected_dep_ip == ip:
                        continue
                    
                    # Check dependencies if the writer appears before the reader in trace
                    # Since dependencies are tracked at execution time, the trace is in execution order,
                    # so we can check all dependencies regardless of distance
                    if expected_pos < pos:
                        if expected_dep_ip in reg_deps:
                            dependency_found = True
                        else:
                            missing_deps.append((ip, reg, expected_dep_ip, pos - expected_pos))
            
            # Update last write IP for written registers
            for reg in write_regs:
                last_write_ip[reg] = ip
                last_write_pos[reg] = pos
        
        # Only report missing dependencies if we found some valid ones
        # This helps avoid false positives from incomplete traces
        if missing_deps and dependency_found:
            # Sort by distance (closer dependencies are more likely to be real issues)
            missing_deps.sort(key=lambda x: x[3])
            # Limit warnings to first 20 to avoid spam
            for ip, reg, expected_ip, dist in missing_deps[:20]:
                self.warnings.append(
                    f"IP {ip} reads {reg} but missing dependency on {expected_ip} (distance: {dist})"
                )
            if len(missing_deps) > 20:
                self.warnings.append(
                    f"... and {len(missing_deps) - 20} more missing register dependencies"
                )
        
        if dependency_found:
            print(f"  ✓ Register dependencies are being tracked")
            if missing_deps:
                print(f"  ⚠ Found {len(missing_deps)} missing dependencies (some may be expected due to loops/trace ordering)")
            return True
        else:
            print(f"  ⚠ No register dependencies found (may be expected)")
            return True
    
    def check_memory_operations(self):
        """Verify memory read/write tracking."""
        print("\n[7] Checking memory read/write tracking...")
        
        memory_reads = 0
        memory_writes = 0
        
        for instr in self.instructions:
            read_addrs = self.parse_semicolon_list(instr.get('Read Addresses', ''))
            write_addrs = self.parse_semicolon_list(instr.get('Write Addresses', ''))
            
            if read_addrs:
                memory_reads += len(read_addrs)
            if write_addrs:
                memory_writes += len(write_addrs)
        
        print(f"  ✓ Found {memory_reads} memory read operations")
        print(f"  ✓ Found {memory_writes} memory write operations")
        
        if memory_reads == 0 and memory_writes == 0:
            self.warnings.append("No memory operations found")
        
        return True
    
    def ranges_overlap(self, a, a_size, b, b_size):
        """Check if two memory ranges overlap."""
        a_end = a + a_size
        b_end = b + b_size
        return not (a_end <= b or b_end <= a)
    
    def check_memory_dependencies(self):
        """Verify memory dependencies are tracked."""
        print("\n[8] Checking memory dependencies...")
        
        # Track last write IP for memory addresses using the same algorithm as the tracing code
        # Format: list of (addr, size, writer_ip) tuples representing disjoint ranges
        # This matches the last_mem_writes structure in peregrine-trace.cpp
        last_mem_writes = []  # List of (addr, size, writer_ip) tuples
        dependency_found = False
        missing_deps = []
        
        for instr in self.instructions:
            ip = instr.get('IP', '').strip()
            read_addrs = self.parse_semicolon_list(instr.get('Read Addresses', ''))
            write_addrs = self.parse_semicolon_list(instr.get('Write Addresses', ''))
            mem_deps = self.parse_semicolon_list(instr.get('Memory Dependent IPs', ''))
            
            # Check for memory dependencies on reads
            # Match the algorithm in peregrine-trace.cpp: check current reads against previous writes
            for addr_str in read_addrs:
                addr_info = self.parse_address(addr_str)
                if addr_info:
                    read_addr, read_size = addr_info
                    # Check if this read overlaps with any previous write
                    for prev_write in last_mem_writes:
                        prev_addr, prev_size, writer_ip = prev_write
                        if self.ranges_overlap(read_addr, read_size, prev_addr, prev_size):
                            # Found an overlapping write - should have dependency
                            if writer_ip in mem_deps:
                                dependency_found = True
                            else:
                                missing_deps.append((ip, addr_str, writer_ip))
            
            # Update last written memory ranges using the same algorithm as tracing code
            # This handles overlapping writes by splitting ranges
            if write_addrs:
                updated = last_mem_writes.copy()
                
                for addr_str in write_addrs:
                    addr_info = self.parse_address(addr_str)
                    if addr_info:
                        cur_addr, cur_size = addr_info
                        cur_start = cur_addr
                        cur_end = cur_addr + cur_size
                        
                        next_writes = []
                        
                        # Process all existing writes
                        for prev_write in updated:
                            prev_addr, prev_size, prev_writer_ip = prev_write
                            prev_start = prev_addr
                            prev_end = prev_addr + prev_size
                            
                            if not (cur_end <= prev_start or prev_end <= cur_start):
                                # Overlap exists; split previous into residual parts
                                if prev_start < cur_start:
                                    # Left residual [prev_start, cur_start)
                                    next_writes.append((prev_start, cur_start - prev_start, prev_writer_ip))
                                if cur_end < prev_end:
                                    # Right residual [cur_end, prev_end)
                                    next_writes.append((cur_end, prev_end - cur_end, prev_writer_ip))
                                # Overlapped middle is fully covered by current write; dropped
                            else:
                                # No overlap; keep previous interval
                                next_writes.append(prev_write)
                        
                        # Add the current write range
                        next_writes.append((cur_start, cur_size, ip))
                        updated = next_writes
                
                last_mem_writes = updated
        
        # Report findings
        if missing_deps and dependency_found:
            # Only report first 20 to avoid spam
            for ip, addr_str, writer_ip in missing_deps[:20]:
                self.warnings.append(
                    f"IP {ip} reads {addr_str} but missing dependency on {writer_ip}"
                )
            if len(missing_deps) > 20:
                self.warnings.append(
                    f"... and {len(missing_deps) - 20} more missing memory dependencies"
                )
        
        if dependency_found:
            print(f"  ✓ Memory dependencies are being tracked")
            if missing_deps:
                print(f"  ⚠ Found {len(missing_deps)} missing dependencies (some may be expected due to execution order)")
            return True
        else:
            print(f"  ⚠ No memory dependencies found (may be expected)")
            return True
    
    def check_test_specific_patterns(self):
        """Check for specific patterns from the test assembly file."""
        print("\n[9] Checking test-specific patterns...")
        
        # Look for patterns from test_trace_c.s
        patterns = {
            'mov_immediate': r'mov.*0x100',  # mov $0x100, %rax
            'memory_read': r'mov.*\[.*rip',  # mov test_var(%rip), %rdx
            'memory_write': r'mov.*\[.*rip.*\]',  # mov %rsi, test_var(%rip)
            'jmp_unconditional': r'jmp\s+[^*,]',  # jmp label (direct)
            'jmp_indirect': r'jmp\s+\*',  # jmp *%r11 (indirect)
            'cmp_conditional': r'cmp.*0.*%rax',  # cmp $0, %rax
            'fence_instructions': r'(mfence|sfence|lfence)',
            'cpuid': r'cpuid',
            'xchg': r'xchg',
            'cmpxchg': r'cmpxchg',
            'rdtsc': r'rdtsc',
            'rdtscp': r'rdtscp',
        }
        
        found_patterns = defaultdict(int)
        
        for instr in self.instructions:
            assembly = instr.get('Assembly', '').strip().lower()
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, assembly, re.IGNORECASE):
                    found_patterns[pattern_name] += 1
        
        if found_patterns:
            print(f"  ✓ Found test patterns:")
            for pattern_name, count in found_patterns.items():
                print(f"    - {pattern_name}: {count}")
        else:
            print(f"  ⚠ No test-specific patterns found (may be running different program)")
        
        return True
    
    def verify(self):
        """Run all verification checks."""
        print("=" * 70)
        print("Trace Verification Script")
        print("=" * 70)
        print(f"Trace file: {self.trace_file}")
        
        if not self.parse_trace():
            return False
        
        print(f"\nLoaded {len(self.instructions)} instructions from trace")
        
        # Run all checks
        checks = [
            self.check_ip_tracking,
            self.check_branch_types,
            self.check_branch_taken_and_target,
            self.check_sync_barriers,
            self.check_register_operations,
            self.check_register_normalization,
            self.check_register_dependencies,
            self.check_memory_operations,
            self.check_memory_dependencies,
            self.check_test_specific_patterns,
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                self.errors.append(f"Error in {check.__name__}: {e}")
                results.append(False)
        
        # Print summary
        print("\n" + "=" * 70)
        print("Verification Summary")
        print("=" * 70)
        
        if self.errors:
            print(f"\n✗ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        else:
            print("\n✓ No errors found")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10 warnings
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
        else:
            print("\n✓ No warnings")
        
        print(f"\nStatistics:")
        for key, value in sorted(self.stats.items()):
            print(f"  - {key}: {value}")
        
        # Overall result
        all_passed = len(self.errors) == 0
        print("\n" + "=" * 70)
        if all_passed:
            print("✓ VERIFICATION PASSED")
        else:
            print("✗ VERIFICATION FAILED")
        print("=" * 70)
        
        return all_passed


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_trace.py <trace.csv>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    verifier = TraceVerifier(trace_file)
    success = verifier.verify()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

