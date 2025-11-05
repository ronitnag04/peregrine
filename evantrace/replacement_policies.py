from enum import Enum
import numpy as np
from numpy.typing import NDArray

from evantrace.utils import where

"""
Class of methods implementing common cache replacement policies
with a unified API. Each method MUST take the affected cache set's tags,
valid bits, policy-specific metadata, set size, and requested tag
as arguments. Each method must return a tuple of hit? (bool), evict? (bool),
and the set index of the accessed/filled/evicted line. Each method must
modify its policy metadata in-place as needed, but may not modify any other
cache variables, this is implemented in the Cache class.
"""
class ReplacementPolicy:
    
    """
    Implements a least recently used replacement policy.
    """
    @staticmethod
    def LRU(
        tags: NDArray[np.uint64],
        valid: NDArray[np.bool],
        metadata: NDArray[np.uint32],
        set_size: np.uint32,
        tag: np.uint64
    ):
        idx = where(tags, tag)
        if idx is not None and valid[idx]:
            # On hit:
            # Set MRU counter for hit line to 0 and
            # increment all lines with lower counter by 1
            hit = True
            evict = False
            indices = np.where(metadata <= metadata[idx])[0]
            metadata[indices] += 1
            metadata[idx] = 0
        else:
            # On miss:
            # Use the first invalid line if possible. If set is full,
            # evict index where LRU counter is equal to set_size.
            # In both cases, set counter of new entry to 0 and
            # increment all other counters
            hit = False
            metadata += 1
            idx = where(valid, np.bool(False))
            if idx is not None:
                evict = False
            else:
                idx = where(metadata, set_size)
                if idx is None:
                    raise ValueError("Could not find max LRU counter in a full set.")
                evict = True
            metadata[idx] = 0 
        return hit, evict, idx
    
    """
    Implements a first-in, first-out replacement policy.
    """
    @staticmethod
    def FIFO(
        tags: NDArray[np.uint64],
        valid: NDArray[np.bool],
        metadata: NDArray[np.uint32],
        set_size: np.uint32,
        tag: np.uint64
    ):
        idx = where(tags, tag)
        if idx is not None and valid[idx]:
            # On hit:
            # Change nothing about metadata
            hit = True
            evict = False
        else:
            # On miss:
            # Use the first invalid line if possible. If set is full,
            # evict index where FIFO counter is equal to set_size.
            # In both cases, set counter of new entry to 0 and
            # increment all other counters.
            hit = False
            metadata += 1
            idx = where(valid, np.bool(False))
            if idx is not None:
                evict = False
            else:
                idx = where(metadata, set_size)
                if idx is None:
                    raise ValueError("Could not find max FIFO counter in a full set.")
                evict = True
            metadata[idx] = 0
        return hit, evict, idx
    
    """
    Implements a tree-based pseudo least recently used cache replacement policy. This
    one can be confusing to follow, see https://en.wikipedia.org/wiki/Pseudo-LRU for
    a reasonably detailed overview.
    """
    @staticmethod
    def PLRU(
        tags: NDArray[np.uint64],
        valid: NDArray[np.bool],
        metadata: NDArray[np.uint32],
        set_size: np.uint32,
        tag: np.uint64
    ):
        if set_size > 32:
            raise ValueError("Set size must be less than or equal to 32 for tree pLRU.")
            
        tree_bits = metadata[0]
        tree_idx = 0 # always start at root
        level_size = set_size
        idx = where(tags, tag)
        if idx is not None and valid[idx]:
            hit = True
            evict = False
        else:
            hit = False
            idx = where(valid, np.bool(False))
            if idx is not None:
                evict = False
            else:
                evict = True
            
        if evict:
            # On miss with full set, traverse tree to pLRU index
            # flipping traversed bits to set new entry to recently used
            idx = 0
            while level_size > 1:
                bit = (tree_bits >> tree_idx) & 1
                tree_bits ^= (1 << tree_idx) # flip visited bit
                if bit == 1:
                    # Go right
                    idx = idx * 2 + 1
                    tree_idx = tree_idx * 2 + 2
                else:
                    # Go left
                    idx *= 2
                    tree_idx = tree_idx * 2 + 1
                level_size //= 2
        else:
            # On hit or miss with space in cache, calculate path through
            # tree to the index and flip bits on path
            assert idx is not None
            
            cur_idx = idx
            num_levels = sum([x if 2**x == set_size else 0 for x in range(10)]) # hacky log2
            for _ in range(num_levels):
                midpoint = level_size // 2
                if cur_idx < midpoint:
                    # Go left
                    tree_bits |= (1 << tree_idx) # point decision bit right
                    tree_idx = 2 * tree_idx + 1
                else:
                    # Go right
                    if tree_bits & (1 << tree_idx) != 0:  # if 1 force to 0
                        tree_bits -= (1 << tree_idx)
                    tree_idx = 2 * tree_idx + 2
                    cur_idx -= midpoint
                level_size = midpoint
                
        metadata[0] = tree_bits 
        return hit, evict, idx
