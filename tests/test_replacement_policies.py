import unittest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from evantrace.replacement_policies import ReplacementPolicy

class TestLRU(unittest.TestCase):
    """
    Tests the ReplacementPolicy.LRU static method with its expanded scope.
    """
    
    # Add type hints for instance variables initialized in setUp
    # to satisfy linters and type checkers.
    set_size: int
    tags: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    metadata: NDArray[np.uint32]

    def __init__(self, *args, **kwargs):
        """
        Initialize attributes to default "empty" values to satisfy linters.
        These will be correctly populated by setUp() before each test.
        """
        super().__init__(*args, **kwargs)
        self.set_size = 0
        self.tags = np.array([], dtype=np.uint64)
        self.valid = np.array([], dtype=np.bool_)
        self.metadata = np.array([], dtype=np.uint32)

    def setUp(self):
        """Common setup for 4-way set associative cache"""
        self.set_size = 4
        # Initial state: Full set with a perfect LRU order
        # Line 0: LRU (age 3)
        # Line 1: (age 2)
        # Line 2: (age 1)
        # Line 3: MRU (age 0)
        self.tags = np.array([10, 20, 30, 40], dtype=np.uint64)
        self.valid = np.array([True, True, True, True], dtype=np.bool_)
        self.metadata = np.array([3, 2, 1, 0], dtype=np.uint32)

    def test_cache_hit_on_mru(self):
        """
        Tests a hit on the Most Recently Used line (age 0).
        - No metadata should change.
        """
        tag_to_access = np.uint64(40) # Accessing line 3
        
        expected_return = (True, False, 3) # (hit, evict, idx)
        expected_metadata = np.array([3, 2, 1, 0], dtype=np.uint32)
        
        result = ReplacementPolicy.LRU(
            self.tags, self.valid, self.metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(self.metadata, expected_metadata, "Metadata changed on MRU hit")

    def test_cache_hit_on_lru(self):
        """
        Tests a hit on the Least Recently Used line (age 3).
        - This line should become MRU (age 0).
        - All other lines should be incremented.
        """
        tag_to_access = np.uint64(10) # Accessing line 0
        
        expected_return = (True, False, 0) # (hit, evict, idx)
        # [3, 2, 1, 0] -> [0, 3, 2, 1]
        # old_age = 3. Lines 1,2,3 (ages 2,1,0) are < 3 and increment.
        expected_metadata = np.array([0, 3, 2, 1], dtype=np.uint32)
        
        result = ReplacementPolicy.LRU(
            self.tags, self.valid, self.metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(self.metadata, expected_metadata, "Metadata not updated correctly on LRU hit")

    def test_cache_hit_on_middle_line(self):
        """
        Tests a hit on a line in the middle of the LRU order (age 2).
        - This line becomes MRU (age 0).
        - Lines *younger* than it (ages 1, 0) are incremented.
        - Lines *older* than it (age 3) are unchanged.
        """
        tag_to_access = np.uint64(20) # Accessing line 1 (age 2)
        
        expected_return = (True, False, 1) # (hit, evict, idx)
        # [3, 2, 1, 0] -> [3, 0, 2, 1]
        # old_age = 2. Lines 2,3 (ages 1,0) are < 2 and increment.
        expected_metadata = np.array([3, 0, 2, 1], dtype=np.uint32)
        
        result = ReplacementPolicy.LRU(
            self.tags, self.valid, self.metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(self.metadata, expected_metadata, "Metadata not updated correctly on middle hit")

    def test_cache_miss_set_not_full(self):
        """
        Tests a miss when the set is not full.
        - Fills the first invalid line.
        - New line becomes MRU (age 0).
        - All *other valid* lines are incremented.
        """
        # State: Only two lines are valid
        tags = np.array([10, 20, 0, 0], dtype=np.uint64)
        valid = np.array([True, True, False, False], dtype=np.bool_)
        metadata = np.array([1, 0, 99, 99], dtype=np.uint32) # Ages 1, 0, and two invalid
        
        tag_to_access = np.uint64(50) # New tag
        
        expected_return = (False, False, 2) # (miss, no evict, idx=2)
        
        # [1, 0] (valid) increment to [2, 1]. New line at idx 2 gets 0.
        expected_metadata = np.array([2, 1, 0, 100], dtype=np.uint32)
        
        result = ReplacementPolicy.LRU(
            tags, valid, metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        # Assertions for tags and valid arrays removed, as they are no longer modified.
        assert_array_equal(metadata, expected_metadata, "Metadata not updated correctly on miss/fill")

    def test_cache_miss_set_full_eviction(self):
        """
        Tests a miss when the set is full, forcing an eviction.
        - Evicts the LRU line (age 3).
        - New line replaces it and becomes MRU (age 0).
        - All other lines are incremented.
        """
        tag_to_access = np.uint64(50) # New tag
        
        # Victim should be line 0 (age 3)
        expected_return = (False, True, 0) # (miss, evict, idx=0)
        
        # [3, 2, 1, 0] -> all increment -> [4, 3, 2, 1]
        # Then victim line 0 is set to 0 -> [0, 3, 2, 1]
        expected_metadata = np.array([0, 3, 2, 1], dtype=np.uint32)

        result = ReplacementPolicy.LRU(
            self.tags, self.valid, self.metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        # Assertions for tags and valid arrays removed, as they are no longer modified.
        assert_array_equal(self.metadata, expected_metadata, "Metadata not updated correctly on miss/evict")
    
    def test_cache_miss_fills_first_invalid(self):
        """
        Tests that a miss fills the *first* invalid line by index.
        """
        tags = np.array([10, 0, 30, 40], dtype=np.uint64)
        valid = np.array([True, False, True, True], dtype=np.bool_)
        metadata = np.array([2, 99, 1, 0], dtype=np.uint32) # Ages 2, 1, 0
        
        tag_to_access = np.uint64(50) # New tag
        
        # Fills line 1, not line 0 (which is valid)
        expected_return = (False, False, 1) # (miss, no evict, idx=1)
        
        # [2, 1, 0] (valid) increment to [3, 2, 1]. New line at idx 1 gets 0.
        expected_metadata = np.array([3, 0, 2, 1], dtype=np.uint32)

        result = ReplacementPolicy.LRU(
            tags, valid, metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        # Assertions for tags and valid arrays removed, as they are no longer modified.
        assert_array_equal(metadata, expected_metadata)
        
class TestFIFO(unittest.TestCase):
    """
    Tests the ReplacementPolicy.FIFO static method with the expanded API.
    (tags, valid, metadata, set_size, tag) -> (hit, evict, idx)
    """
    
    # Type hints for instance variables
    set_size: int
    tags: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    metadata: NDArray[np.uint32]

    def __init__(self, *args, **kwargs):
        """
        Initialize attributes to default "empty" values to satisfy linters.
        These will be correctly populated by setUp() before each test.
        """
        super().__init__(*args, **kwargs)
        self.set_size = 0
        self.tags = np.array([], dtype=np.uint64)
        self.valid = np.array([], dtype=np.bool_)
        self.metadata = np.array([], dtype=np.uint32)

    def setUp(self):
        """Common setup for 4-way set associative cache"""
        self.set_size = 4
        # Initial state: Full set with a perfect FIFO order
        # Line 0: Oldest (age 3) - Victim
        # Line 1: (age 2)
        # Line 2: (age 1)
        # Line 3: Newest (age 0)
        self.tags = np.array([10, 20, 30, 40], dtype=np.uint64)
        self.valid = np.array([True, True, True, True], dtype=np.bool_)
        self.metadata = np.array([3, 2, 1, 0], dtype=np.uint32)

    def test_cache_hit(self):
        """
        Tests a hit on any line.
        - For FIFO, metadata should NOT change on a hit.
        """
        tag_to_access = np.uint64(20) # Accessing line 1 (age 2)
        
        expected_return = (True, False, 1) # (hit, no evict, idx=1)
        # Metadata does not change on a FIFO hit
        expected_metadata = np.array([3, 2, 1, 0], dtype=np.uint32)
        
        # Create a copy to ensure the method doesn't modify it in place
        metadata_copy = self.metadata.copy()
        
        result = ReplacementPolicy.FIFO(
            self.tags, self.valid, metadata_copy, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata_copy, expected_metadata, "Metadata changed on FIFO hit")

    def test_cache_miss_set_not_full(self):
        """
        Tests a miss when the set is not full.
        - Fills the first invalid line.
        - New line becomes newest (age 0).
        - All *other valid* lines are incremented.
        """
        # State: Only two lines are valid
        tags = np.array([10, 20, 0, 0], dtype=np.uint64)
        valid = np.array([True, True, False, False], dtype=np.bool_)
        metadata = np.array([1, 0, 99, 99], dtype=np.uint32) # Ages 1, 0, and two invalid
        
        tag_to_access = np.uint64(50) # New tag
        
        expected_return = (False, False, 2) # (miss, no evict, idx=2)
        
        # [1, 0] (valid) increment to [2, 1]. New line at idx 2 gets 0.
        expected_metadata = np.array([2, 1, 0, 100], dtype=np.uint32)
        
        result = ReplacementPolicy.FIFO(
            tags, valid, metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata, expected_metadata, "Metadata not updated correctly on miss/fill")

    def test_cache_miss_set_full_eviction(self):
        """
        Tests a miss when the set is full, forcing an eviction.
        - Evicts the oldest line (age 3).
        - New line replaces it and becomes newest (age 0).
        - All other lines are incremented.
        """
        tag_to_access = np.uint64(50) # New tag
        
        # Victim should be line 0 (age 3)
        expected_return = (False, True, 0) # (miss, evict, idx=0)
        
        # [3, 2, 1, 0] -> all increment -> [4, 3, 2, 1]
        # Then victim line 0 is set to 0 -> [0, 3, 2, 1]
        expected_metadata = np.array([0, 3, 2, 1], dtype=np.uint32)

        # Use a copy since setUp metadata is modified in-place
        metadata_copy = self.metadata.copy()

        result = ReplacementPolicy.FIFO(
            self.tags, self.valid, metadata_copy, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata_copy, expected_metadata, "Metadata not updated correctly on miss/evict")
    
    def test_cache_miss_fills_first_invalid(self):
        """
        Tests that a miss fills the *first* invalid line by index.
        """
        tags = np.array([10, 0, 30, 40], dtype=np.uint64)
        valid = np.array([True, False, True, True], dtype=np.bool_)
        metadata = np.array([2, 99, 1, 0], dtype=np.uint32) # Ages 2, 1, 0
        
        tag_to_access = np.uint64(50) # New tag
        
        # Fills line 1, not line 0 (which is valid)
        expected_return = (False, False, 1) # (miss, no evict, idx=1)
        
        # [2, 1, 0] (valid) increment to [3, 2, 1]. New line at idx 1 gets 0.
        expected_metadata = np.array([3, 0, 2, 1], dtype=np.uint32)

        result = ReplacementPolicy.FIFO(
            tags, valid, metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata, expected_metadata)
        
class TestPLRU(unittest.TestCase):
    """
    Tests the ReplacementPolicy.PLRU static method with the expanded API.
    (tags, valid, metadata, set_size, tag) -> (hit, evict, idx)
    
    Assumes a 4-way set for testing.
    Tree:
        bit 0 (root)
       /     \
    bit 1   bit 2
    /   \\  /   \
    L0  L1 L2  L3
    
    Path L->L (bits 0, 1) -> Line 0
    Path L->R (bits 0, 1) -> Line 1
    Path R->L (bits 0, 2) -> Line 2
    Path R->R (bits 0, 2) -> Line 3
    
    Update on hit/fill (line X): Set bits on path to X to point *away* from X.
    """
    
    # Type hints for instance variables
    set_size: int
    tags: NDArray[np.uint64]
    valid: NDArray[np.bool_]
    metadata: NDArray[np.uint32]

    def __init__(self, *args, **kwargs):
        """
        Initialize attributes to default "empty" values to satisfy linters.
        """
        super().__init__(*args, **kwargs)
        self.set_size = 0
        self.tags = np.array([], dtype=np.uint64)
        self.valid = np.array([], dtype=np.bool_)
        # Note: PLRU metadata only uses one element, but we'll
        # initialize a small array for type consistency.
        self.metadata = np.array([0], dtype=np.uint32) 

    def setUp(self):
        """Common setup for 4-way set associative cache"""
        self.set_size = 4
        self.tags = np.array([10, 20, 30, 40], dtype=np.uint64)
        self.valid = np.array([True, True, True, True], dtype=np.bool_)
        # Initial state: 0b000.
        # Victim path: L (bit 0=0) -> L (bit 1=0) -> Victim = Line 0
        self.metadata = np.array([0b000], dtype=np.uint32)

    def test_cache_hit(self):
        """
        Tests a hit on a line (e.g., Line 2).
        - Metadata should be updated to make Line 2 MRU.
        """
        tag_to_access = np.uint64(30) # Accessing line 2
        
        expected_return = (True, False, 2) # (hit, no evict, idx=2)
        
        # Path to Line 2 is R->L (bits 0, 2).
        # Initial bits: 0b000
        # 1. Path R (bit 0): Set bit 0 to point L (0). Bits remain 0b000.
        # 2. Path L (bit 2): Set bit 2 to point R (1). Bits become 0b100.
        expected_metadata = np.array([0b100], dtype=np.uint32)
        
        # Use a copy since metadata is modified in-place
        metadata_copy = self.metadata.copy()
        
        result = ReplacementPolicy.PLRU(
            self.tags, self.valid, metadata_copy, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata_copy, expected_metadata, "Metadata not updated correctly on hit")

    def test_cache_miss_set_not_full(self):
        """
        Tests a miss when the set is not full.
        - Fills the first invalid line (idx 2).
        - New line (idx 2) becomes MRU.
        """
        # State: Only two lines are valid
        tags = np.array([10, 20, 0, 0], dtype=np.uint64)
        valid = np.array([True, True, False, False], dtype=np.bool_)
        # Initial state: 0b000. Victim would be Line 0, but we fill first.
        metadata = np.array([0b000], dtype=np.uint32) 
        
        tag_to_access = np.uint64(50) # New tag
        
        # Fills first invalid line, idx=2
        expected_return = (False, False, 2) # (miss, no evict, idx=2)
        
        # Update bits to make Line 2 MRU (same logic as test_cache_hit)
        # Path R->L (bits 0, 2). Bits 0b000 -> 0b100.
        expected_metadata = np.array([0b100], dtype=np.uint32)
        
        result = ReplacementPolicy.PLRU(
            tags, valid, metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata, expected_metadata, "Metadata not updated correctly on miss/fill")

    def test_cache_miss_set_full_eviction(self):
        """
        Tests a miss when the set is full, forcing an eviction.
        - Evicts the PLRU victim (Line 0).
        - New line replaces it and becomes MRU (Line 0).
        """
        tag_to_access = np.uint64(50) # New tag
        
        # Initial state 0b000 -> Victim is Line 0
        expected_return = (False, True, 0) # (miss, evict, idx=0)
        
        # Update bits to make Line 0 MRU.
        # Path to Line 0 is L->L (bits 0, 1).
        # Initial bits: 0b000
        # 1. Path L (bit 0): Set bit 0 to point R (1). Bits become 0b001.
        # 2. Path L (bit 1): Set bit 1 to point R (1). Bits become 0b011.
        expected_metadata = np.array([0b011], dtype=np.uint32)

        # Use a copy since setUp metadata is modified in-place
        metadata_copy = self.metadata.copy()

        result = ReplacementPolicy.PLRU(
            self.tags, self.valid, metadata_copy, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata_copy, expected_metadata, "Metadata not updated correctly on miss/evict")
    
    def test_cache_miss_fills_first_invalid(self):
        """
        Tests that a miss fills the *first* invalid line by index,
        even if it's not the PLRU victim.
        """
        tags = np.array([10, 0, 30, 40], dtype=np.uint64)
        valid = np.array([True, False, True, True], dtype=np.bool_)
        # Initial state 0b000 -> Victim would be Line 0, but we fill Line 1
        metadata = np.array([0b000], dtype=np.uint32) 
        
        tag_to_access = np.uint64(50) # New tag
        
        # Fills line 1 (the first invalid line)
        expected_return = (False, False, 1) # (miss, no evict, idx=1)
        
        # Update bits to make Line 1 MRU.
        # Path to Line 1 is L->R (bits 0, 1).
        # Initial bits: 0b000
        # 1. Path L (bit 0): Set bit 0 to point R (1). Bits become 0b001.
        # 2. Path R (bit 1): Set bit 1 to point L (0). Bits remain 0b001.
        expected_metadata = np.array([0b001], dtype=np.uint32)

        result = ReplacementPolicy.PLRU(
            tags, valid, metadata, self.set_size, tag_to_access
        )
        
        self.assertTupleEqual(result, expected_return)
        assert_array_equal(metadata, expected_metadata)

def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLRU))
    suite.addTests(loader.loadTestsFromTestCase(TestFIFO))
    suite.addTests(loader.loadTestsFromTestCase(TestPLRU))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*70}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)