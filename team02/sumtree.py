import numpy as np

class SumTree:
    """Binary sum tree for efficient sampling based on priorities"""
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree array (internal nodes + leaves)
        self.data = np.zeros(capacity, dtype=object)  # Data array (only leaves)
        self.size = 0  # Current size
        self.next_idx = 0  # Next index to write to
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):  # If we're at a leaf
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Return total priority"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add data with priority to tree"""
        # Store data in leaf node
        tree_idx = self.next_idx + self.capacity - 1
        self.data[self.next_idx] = data
        
        # Update tree with priority
        self.update(tree_idx, priority)
        
        # Update indices
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update priority of existing data"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get an experience with priority s"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]