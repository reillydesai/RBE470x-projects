import random
import numpy as np
import torch
from sumtree import SumTree

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay implementation"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Priority exponent - controls sampling bias
        self.beta = beta  # Initial importance sampling weight
        self.beta_increment = beta_increment  # Beta annealing rate
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Max priority starts at 1
    
    def __len__(self):
        return self.tree.size
    
    def add(self, experience, error=None):
        """Add experience to buffer with priority based on TD error"""
        # Default max priority for new experiences (ensures they get sampled at least once)
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """Sample batch_size experiences with prioritization"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # Increment beta toward 1 over time (reduces bias)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Get a sample from each segment
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (len(self) * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize to max weight = 1
        
        return batch, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, errors):
        """Update priorities after learning"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)