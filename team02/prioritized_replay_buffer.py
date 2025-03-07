import numpy as np
import random
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha  # Controls the degree of prioritization

    def push(self, experience):
        """Store experience with priority"""
        priority = self.calculate_priority(experience)  # Ensure this returns a scalar value
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)  # Ensure priority is a scalar
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def calculate_priority(self, experience):
        """Calculate priority based on TD error or other criteria"""
        # Ensure this returns a scalar value
        return abs(experience[2]) + 1e-5  # Ensure priority is always positive

    def sample(self, batch_size):
        """Sample experiences based on priority"""
        if len(self.buffer) == 0:
            return [], [], []

        # Calculate probabilities based on priorities
        priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha  # Ensure dtype is consistent
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-1)  # Adjust as needed
        weights /= weights.max()  # Normalize weights

        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority 