from typing import Callable
import numpy as np
from numpy.typing import NDArray

from evantrace.replacement_policies import ReplacementPolicy

MAIN_MEMORY_LATENCY = 100 # latency of main memory accesses in cycles

"""
Representation of a hardware cache, with configurable line size, total size, associativity,
replacement policy, address space, read latency, write latency. May also be configured as
part of multi-level cache system.
"""
class Cache:
    
    def __init__(
        self,
        associativity: int = 1,     # direct mapped
        line_size: int = 64,        # 64 bytes
        total_size: int = 16000,    # 16 kB
        replacement_policy: Callable[[NDArray[np.uint64], NDArray[np.bool], NDArray[np.uint32], np.uint32, np.uint64], tuple[bool, bool, int]] = ReplacementPolicy.LRU,
        address_bits: int = 64,     # 64 bit address space
        read_latency: int = 4,      # 4 cycles
        write_latency: int = 0,     # assume this is masked by a store buffer
        parent: Cache | None = None # parent cache to send misses and write back to, if None, assumes a main memory access
    ):
        # Check and initialize sizing
        if address_bits & (address_bits - 1) != 0:
            raise ValueError("Number of address bits must be a power of 2.")
        self.address_bits: int = address_bits
        if associativity < 1 or associativity & (associativity - 1) != 0:
            raise ValueError("Associativity must be a power of 2.")
        self.associativity: np.uint32 = np.uint32(associativity)
        if line_size < 1 or line_size & (line_size - 1) != 0:
            raise ValueError("Line size must be a power of 2.")
        self.line_size: int = line_size
        self.total_size: int = total_size
        if total_size % line_size != 0:
            raise ValueError("Total cache size must be cleanly divisible by line size.")
        self.num_lines: int = total_size // line_size
        if self.num_lines % associativity != 0:
            raise ValueError("Number of lines must be cleanly divisible by associativity.")
        self.num_sets: int = self.num_lines // associativity
        
        # Calculate tag, index, and offset bits
        self.offset_bits: int = np.log2(line_size)
        self.index_bits: int = np.log2(self.num_sets)
        self.tag_bits: int = address_bits - self.index_bits - self.offset_bits
        
        # tag, valid, and dirty bits per line
        self.tags: list[NDArray[np.uint64]] = [np.zeros((associativity,), dtype=np.uint64)] * self.num_sets
        self.valid: list[NDArray[np.bool]] = [np.zeros((associativity,), dtype=np.bool)] * self.num_sets
        self.dirty: list[NDArray[np.bool]] = [np.zeros((associativity,), dtype=np.bool)] * self.num_sets
        
        # Replacement policy related
        self.replacement_policy: Callable[[NDArray[np.uint64], NDArray[np.bool], NDArray[np.uint32], np.uint32, np.uint64], tuple[bool, bool, int]] = replacement_policy
        self.metadata: list[NDArray[np.uint32]] = [np.zeros((associativity,), dtype=np.uint32)] * self.num_sets
        
        # Latency related
        self.read_latency: int = read_latency
        self.write_latency: int = write_latency
        if parent is not None and parent.address_bits != address_bits:
            raise ValueError("Caches in the same system must have the same number of address bits")
        self.parent: Cache | None = parent
    
    """
    Simulates a cache read using the configured replacement policy, forwarding requests to
    higher level caches if needed. Returns total read latency for given address.
    """
    def read(self, address: np.uint64) -> int:
        offset = address & (2**self.offset_bits - 1)
        index = (address >> self.offset_bits) & (2**self.index_bits - 1)
        tag = (address >> (self.index_bits + self.offset_bits)) # should be the rest of the address
        
        if self.associativity == 1:
            # No replacement policy needed for a direct mapped cache
            hit = self.tags[index][0] == tag and self.valid[index][0]
            evict = not hit and self.valid[index][0]
            set_idx = 0
        else:
            # Apply replacement policy
            hit, evict, set_idx = self.replacement_policy(
                self.tags[index],
                self.valid[index],
                self.metadata[index],
                self.associativity,
                tag
            )
            
        if hit:
            # On a hit, we simply return the cache's own read latency
            # without needing to update any state
            return self.read_latency
        
        if not evict or not self.dirty[index][set_idx]:
            # On a miss without eviction or a miss with eviction of a non-dirty line, 
            # we return the cache's own read latency plus the
            # read latency of the parent for the requested line
            # (or simply main memory latency if parent is None)
            if self.parent is None:
                latency = self.read_latency + MAIN_MEMORY_LATENCY
            else:
                latency = self.read_latency + self.parent.read(address)
            self.valid[index][set_idx] = True
        else:
            # On a miss with an eviction of a dirty line, we return the cache's own read latency plus the
            # write latency of the parent for the evicted line plus the read latency
            # of the parent for the requested line
            # (or plus main memory latency if parent is None)
            victim_line = (self.tags[index][set_idx] << (self.index_bits + self.offset_bits)) + (index << self.offset_bits)
            if self.parent is None:
                latency = self.read_latency + MAIN_MEMORY_LATENCY
            else:
                latency = self.read_latency + self.parent.write(victim_line) + self.parent.read(address)
            self.dirty[index][set_idx] = False
        
        # Update tags state before returning latency
        self.tags[index][set_idx] = tag
        return latency
        
    """
    Simulates a cache write using the configured replacement policy, forwarding evictions
    to higher level caches if needed. Returns total write latency for given address.
    """
    def write(self, address: np.uint64) -> int:
        offset = address & (2**self.offset_bits - 1)
        index = (address >> self.offset_bits) & (2**self.index_bits - 1)
        tag = (address >> (self.index_bits + self.offset_bits)) # should be the rest of the address
        
        if self.associativity == 1:
            # No replacement policy needed for a direct mapped cache
            hit = self.tags[index][0] == tag and self.valid[index][0]
            evict = not hit and self.valid[index][0]
            set_idx = 0
        else:
            # Apply replacement policy
            hit, evict, set_idx = self.replacement_policy(
                self.tags[index],
                self.valid[index],
                self.metadata[index],
                self.associativity,
                tag
            )
        
        if hit:
            # On a hit, we simply return the cache's own write latency,
            # only needing to set the dirty bit for the line to True
            latency = self.write_latency
        elif not evict or not self.dirty[index][set_idx]:
            # On a miss without eviction or a miss with eviction of a non-dirty line, 
            # we return the cache's own write latency and update the cache state
            latency = self.write_latency
            self.valid[index][set_idx] = True
        else:
            # On a miss with an eviction of a dirty line, we return the cache's own
            # write latency plus the write latency of the parent for the evicted line
            victim_line = (self.tags[index][set_idx] << (self.index_bits + self.offset_bits)) + (index << self.offset_bits)
            if self.parent is None:
                latency = self.write_latency
            else:
                latency = self.write_latency + self.parent.write(victim_line)
        
        self.dirty[index][set_idx] = True
        self.tags[index][set_idx] = tag
        return latency
            