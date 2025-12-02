"""
Branch predictor implementations for simulating processor branch prediction behavior.
"""
import numpy as np
from abc import ABC, abstractmethod
from evantrace.x86.branch_types import Branch_Type

class BranchPredictor(ABC):
    """
    Abstract base class for branch predictors.
    """
    
    @abstractmethod
    def predict(self, inst_ptr: np.uint64, branch_type: Branch_Type) -> bool:
        """
        Predict whether a branch will be taken.
        
        Args:
            inst_ptr: Instruction pointer (address) of the branch instruction
            branch_type: Type of branch (direct_conditional, direct_unconditional, indirect)
        
        Returns:
            True if branch is predicted taken, False otherwise
        """
        pass
    
    @abstractmethod
    def update(self, inst_ptr: np.uint64, branch_type: Branch_Type, predicted_taken: bool, actual_taken: bool):
        """
        Update the predictor state based on the actual branch outcome.
        
        Args:
            inst_ptr: Instruction pointer (address) of the branch instruction
            branch_type: Type of branch
            predicted_taken: Result returned by predict() for this branch
            actual_taken: True if branch was actually taken, False otherwise
        """
        pass


class SimpleBranchPredictor(BranchPredictor):
    """
    A simple branch predictor that predicts randomly (taken or not taken).
    For unconditional and indirect branches, always predicts taken.
    For conditional branches, predicts randomly with 50% chance of taken/not taken.
    """
    
    def __init__(self, random_seed: int | None = None):
        """
        Initialize the simple branch predictor.
        
        Args:
            random_seed: Optional random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        self.total_predictions = 0
        self.total_mispredictions = 0
    
    def predict(self, inst_ptr: np.uint64, branch_type: Branch_Type) -> bool:
        """
        Predict whether a branch will be taken.
        For unconditional and indirect branches, always predicts taken.
        For conditional branches, predicts randomly (50% chance of taken/not taken).
        
        Args:
            inst_ptr: Instruction pointer (address) of the branch instruction
            branch_type: Type of branch
        
        Returns:
            True if branch is predicted taken, False otherwise
        """
        self.total_predictions += 1
        
        # Unconditional branches are always taken
        if branch_type == Branch_Type.direct_unconditional or branch_type == Branch_Type.indirect:
            return True
        
        # For conditional branches, predict randomly
        return self.rng.random() < 0.5

    def update(self, inst_ptr: np.uint64, branch_type: Branch_Type, predicted_taken: bool, actual_taken: bool):
        """
        Update the predictor state based on the actual branch outcome.
        Tracks misprediction statistics.
        
        Args:
            inst_ptr: Instruction pointer (address) of the branch instruction
            branch_type: Type of branch
            predicted_taken: Result returned by predict() for this branch
            actual_taken: True if branch was actually taken, False otherwise
        """
        if predicted_taken != actual_taken:
            self.total_mispredictions += 1
    
    def get_misprediction_rate(self) -> float:
        """
        Get the actual misprediction rate observed so far.
        
        Returns:
            Misprediction rate (0.0 to 1.0)
        """
        if self.total_predictions == 0:
            return 0.0
        return self.total_mispredictions / self.total_predictions


class TAGEBranchPredictor(BranchPredictor):
    """
    TAGE (TAgged GEometric length) branch predictor implementation.
    TAGE is a state-of-the-art branch predictor that uses multiple tagged
    geometric-length history tables to make predictions.
    """
    
    def __init__(
        self,
        num_tables: int = 4,
        table_sizes: list[int] | None = None,
        history_lengths: list[int] | None = None,
        tag_bits: int = 10,
        counter_bits: int = 2
    ):
        """
        Initialize the TAGE branch predictor.
        
        Args:
            num_tables: Number of prediction tables
            table_sizes: List of sizes for each table. If None, uses geometric progression.
            history_lengths: List of history lengths for each table. If None, uses geometric progression.
            tag_bits: Number of bits for tags
            counter_bits: Number of bits for saturating counters
        """
        self.num_tables = num_tables
        self.tag_bits = tag_bits
        self.counter_bits = counter_bits
        self.max_counter = (1 << counter_bits) - 1
        
        # Initialize table sizes with geometric progression if not provided
        if table_sizes is None:
            base_size = 1024
            self.table_sizes = [base_size * (2 ** i) for i in range(num_tables)]
        else:
            if len(table_sizes) != num_tables:
                raise ValueError("Number of table sizes must match num_tables")
            self.table_sizes = table_sizes
        
        # Initialize history lengths with geometric progression if not provided
        if history_lengths is None:
            self.history_lengths = [2 ** i for i in range(num_tables)]
        else:
            if len(history_lengths) != num_tables:
                raise ValueError("Number of history lengths must match num_tables")
            self.history_lengths = history_lengths
        
        # Initialize prediction tables: each entry has (tag, counter, useful)
        # tag: tag_bits, counter: counter_bits, useful: 1 bit
        self.tables = []
        for size in self.table_sizes:
            table = {
                'tags': np.zeros(size, dtype=np.uint32),
                'counters': np.zeros(size, dtype=np.uint8),  # saturating counters
                'useful': np.zeros(size, dtype=np.bool)  # useful bit for replacement
            }
            self.tables.append(table)
        
        # Global branch history register
        self.global_history = 0
        self.max_history_bits = max(self.history_lengths)
        
        # Base predictor (bimodal) for fallback
        self.bimodal_size = 4096
        self.bimodal = np.zeros(self.bimodal_size, dtype=np.uint8)
        
        # Statistics
        self.total_predictions = 0
        self.total_mispredictions = 0
    
    def _hash_pc(self, inst_ptr: np.uint64, table_idx: int) -> int:
        """Hash instruction pointer for table indexing."""
        return int((inst_ptr ^ (inst_ptr >> np.uint64(table_idx + 1))) % np.uint64(self.table_sizes[table_idx]))
    
    def _hash_tag(self, inst_ptr: np.uint64, history: int) -> np.uint32:
        """Hash instruction pointer and history for tag generation."""
        return np.uint32((inst_ptr ^ np.uint64(history)) & np.uint64((1 << self.tag_bits) - 1))
    
    def _get_history(self, table_idx: int) -> int:
        """Get history for a specific table (masked to appropriate length)."""
        mask = (1 << self.history_lengths[table_idx]) - 1
        return self.global_history & mask
    
    def _predict_internal(self, inst_ptr: np.uint64, branch_type: Branch_Type) -> bool:
        """
        Internal prediction method without incrementing statistics.
        """
        # Unconditional branches are always taken
        if branch_type == Branch_Type.direct_unconditional or branch_type == Branch_Type.indirect:
            return True
        
        # Check TAGE tables from longest history to shortest
        provider_idx = None
        alt_pred = None
        
        for table_idx in range(self.num_tables - 1, -1, -1):
            idx = self._hash_pc(inst_ptr, table_idx)
            history = self._get_history(table_idx)
            tag = self._hash_tag(inst_ptr, history)
            
            table = self.tables[table_idx]
            if table['tags'][idx] == tag:
                # Tag match found
                counter = table['counters'][idx]
                pred = bool(counter >= np.uint8(1 << (self.counter_bits - 1)))  # Threshold for taken
                
                if provider_idx is None:
                    provider_idx = table_idx
                    provider_pred = pred
                else:
                    alt_pred = pred
                    break
        
        # Use provider prediction if found, otherwise use bimodal
        if provider_idx is not None:
            prediction = provider_pred
        else:
            # Fallback to bimodal predictor
            bimodal_idx = int(inst_ptr) % self.bimodal_size
            prediction = bool(self.bimodal[bimodal_idx] >= np.uint8(1 << (self.counter_bits - 1)))
            alt_pred = prediction  # For consistency
        
        return prediction
    
    def predict(self, inst_ptr: np.uint64, branch_type: Branch_Type) -> bool:
        """
        Predict whether a branch will be taken using TAGE algorithm.
        
        Args:
            inst_ptr: Instruction pointer (address) of the branch instruction
            branch_type: Type of branch
        
        Returns:
            True if branch is predicted taken, False otherwise
        """
        self.total_predictions += 1
        return self._predict_internal(inst_ptr, branch_type)
    
    def update(self, inst_ptr: np.uint64, branch_type: Branch_Type, predicted_taken: bool, actual_taken: bool):
        """
        Update the TAGE predictor state based on the actual branch outcome.
        
        Args:
            inst_ptr: Instruction pointer (address) of the branch instruction
            branch_type: Type of branch
            predicted_taken: Result returned by predict() for this branch
            actual_taken: True if branch was actually taken, False otherwise
        """
        if branch_type == Branch_Type.direct_unconditional or branch_type == Branch_Type.indirect:
            return  # No update needed for unconditional branches
        
        # Find provider table
        provider_idx = None
        alt_pred = None
        
        for table_idx in range(self.num_tables - 1, -1, -1):
            idx = self._hash_pc(inst_ptr, table_idx)
            history = self._get_history(table_idx)
            tag = self._hash_tag(inst_ptr, history)
            
            table = self.tables[table_idx]
            if table['tags'][idx] == tag:
                if provider_idx is None:
                    provider_idx = table_idx
                    provider_pred = bool(table['counters'][idx] >= np.uint8(1 << (self.counter_bits - 1)))
                else:
                    alt_pred = bool(table['counters'][idx] >= np.uint8(1 << (self.counter_bits - 1)))
                    break
        
        if predicted_taken != actual_taken:
            self.total_mispredictions += 1
        
        # Update provider table
        if provider_idx is not None:
            idx = self._hash_pc(inst_ptr, provider_idx)
            table = self.tables[provider_idx]
            
            # Update counter (saturating)
            if actual_taken and table['counters'][idx] < np.uint8(self.max_counter):
                table['counters'][idx] = np.uint8(min(int(table['counters'][idx]) + 1, self.max_counter))
            elif not actual_taken and table['counters'][idx] > np.uint8(0):
                table['counters'][idx] = np.uint8(max(int(table['counters'][idx]) - 1, 0))
            
            # Update useful bit
            if alt_pred is not None and provider_pred != alt_pred:
                # Useful if provider was correct and alt was wrong
                if provider_pred == actual_taken and alt_pred != actual_taken:
                    table['useful'][idx] = True
                elif provider_pred != actual_taken and alt_pred == actual_taken:
                    table['useful'][idx] = False
        else:
            # Update bimodal predictor
            bimodal_idx = int(inst_ptr) % self.bimodal_size
            if actual_taken and self.bimodal[bimodal_idx] < np.uint8(self.max_counter):
                self.bimodal[bimodal_idx] = np.uint8(min(int(self.bimodal[bimodal_idx]) + 1, self.max_counter))
            elif not actual_taken and self.bimodal[bimodal_idx] > np.uint8(0):
                self.bimodal[bimodal_idx] = np.uint8(max(int(self.bimodal[bimodal_idx]) - 1, 0))
        
        # Allocate new entry if mispredicted and no provider found
        if predicted_taken != actual_taken and provider_idx is None:
            # Find a table to allocate in (prefer shorter history tables)
            for table_idx in range(self.num_tables):
                idx = self._hash_pc(inst_ptr, table_idx)
                history = self._get_history(table_idx)
                tag = self._hash_tag(inst_ptr, history)
                
                table = self.tables[table_idx]
                if not table['useful'][idx]:
                    # Allocate here
                    table['tags'][idx] = tag
                    table['counters'][idx] = np.uint8((1 << (self.counter_bits - 1)) + (1 if actual_taken else 0))
                    table['useful'][idx] = False
                    break
        
        # Update global history
        self.global_history = ((self.global_history << 1) | (1 if actual_taken else 0)) & ((1 << self.max_history_bits) - 1)
    
    def get_misprediction_rate(self) -> float:
        """
        Get the actual misprediction rate observed so far.
        
        Returns:
            Misprediction rate (0.0 to 1.0)
        """
        if self.total_predictions == 0:
            return 0.0
        return self.total_mispredictions / self.total_predictions

