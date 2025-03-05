# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')

# Import necessary stuff
from entity import CharacterEntity
from events import Event

# Neural network imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class DQNetwork(nn.Module):
    """Neural network for Deep Q-learning"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class TestCharacter(CharacterEntity):
    def __init__(self, name, avatar, x=0, y=0):
        super().__init__(name, avatar, x, y)
        
        # Set absolute path for model save and print it
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.pth")
        print(f"Model path is: {self.model_path}")
        
        # Define all possible actions
        self.all_actions = [
            (0, 0, False), (0, 0, True),    # Stay in place
            (0, 1, False), (0, 1, True),    # Move up
            (0, -1, False), (0, -1, True),  # Move down
            (1, 0, False), (1, 0, True),    # Move right
            (-1, 0, False), (-1, 0, True),  # Move left
            (1, 1, False), (1, 1, True),    # Move up-right
            (-1, -1, False), (-1, -1, True),# Move down-left
            (1, -1, False), (1, -1, True),  # Move down-right
            (-1, 1, False), (-1, 1, True),  # Move up-left
        ]
        
        # DQL Parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Starting exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # State and action dimensions
        self.state_size = 10  # We'll define our state features
        self.action_size = 20  # 10 movements × 2 (bomb or no bomb)
        
        # Neural Networks (main and target)
        self.main_network = DQNetwork(self.state_size, 64, self.action_size)
        self.target_network = DQNetwork(self.state_size, 64, self.action_size)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Additional tracking variables
        self.steps = 0
        self.update_target_every = 100  # Update target network every 100 steps 

        # Load previously trained model if it exists
        if os.path.exists(self.model_path):
            self.load_model()

    def get_state_features(self, wrld):
        """Convert world state into a feature vector"""
        me = wrld.me(self)
        if me is None:
            return None
            
        # Goal is always the bottom-right corner
        goal = (wrld.width() - 1, wrld.height() - 1)
        
        # Get current position features
        x, y = me.x, me.y
        
        # Calculate distances to all monsters
        monster_distances = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.monsters_at(i, j):
                    monster_distances.append(abs(x - i) + abs(y - j))
        
        # Sort monster distances to get closest and second closest
        monster_distances = sorted(monster_distances + [8, 8])  # Pad with 8 if less than 2 monsters
        
        # Count nearby walls that need clearing
        walls_blocking_path = 0
        path_positions = self.get_path_positions((x, y), goal)
        for px, py in path_positions[:5]:  # Check first 5 positions in path
            if wrld.wall_at(px, py):
                walls_blocking_path += 1
        
        # Features vector
        features = [
            x / wrld.width(),  # Normalized x position
            y / wrld.height(),  # Normalized y position
            self.heuristic((x, y), goal) / (wrld.width() + wrld.height()),  # Normalized distance to exit
            monster_distances[0] / 8.0,  # Distance to closest monster
            monster_distances[1] / 8.0,  # Distance to second closest monster
            walls_blocking_path / 5.0,  # Normalized count of walls blocking path
            len(self.get_safe_neighbors(wrld, (x, y))) / 8.0,  # Available safe moves
            1.0 if (x, y) in self.get_explosion_positions(wrld) else 0.0,  # In explosion range
            1.0 if any(wrld.bomb_at(i, j) for i in range(max(0, x-4), min(wrld.width(), x+5))
                                         for j in range(max(0, y-4), min(wrld.height(), y+5))) else 0.0,  # Bomb nearby
            1.0 if self.is_backtracking_needed(wrld, (x, y), goal) else 0.0  # Need to backtrack
        ]
        
        return torch.FloatTensor(features)

    def get_possible_actions(self, wrld, state):
        """Get list of possible actions and their indices"""
        actions = []
        valid_indices = []
        
        for i, (dx, dy, place_bomb) in enumerate(self.all_actions):
            if i < self.action_size and self.is_valid_move(wrld, state, (dx, dy)):
                actions.append(((dx, dy), place_bomb))
                valid_indices.append(i)
                    
        return actions, valid_indices

    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store transition in replay memory"""
        if state is not None:  # Only check if current state is valid
            # For terminal states (done=True), next_state can be None
            if next_state is None:
                next_state = torch.zeros_like(state)  # Zero tensor for terminal states
            self.memory.append((state, action_idx, reward, next_state, done))

    def train_network(self):
        """Train the network using a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([s for s, _, _, _, _ in batch])
        next_states = torch.stack([ns for _, _, _, ns, _ in batch])
        actions = torch.LongTensor([a for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        dones = torch.FloatTensor([d for _, _, _, _, d in batch])
        
        # Get current Q values
        current_q = self.main_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update main network
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Update target network weights"""
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

    def get_reward(self, wrld, state, action, next_state, events):
        """Calculate reward for the given transition"""
        reward = 0
        x, y = next_state
        
        # Base reward based on distance to goal
        goal = (wrld.width() - 1, wrld.height() - 1)
        current_distance = self.heuristic(state, goal)
        next_distance = self.heuristic(next_state, goal)
        reward += (current_distance - next_distance) * 2
        
        # Penalties for being close to monsters
        monster_distance = self.get_closest_monster_distance(wrld, (x, y))
        if monster_distance == 2:
            reward -= 5
        elif monster_distance == 3:
            reward -= 2
        
        # Bomb placement rewards/penalties
        direction, place_bomb = action
        if place_bomb:
            # Count walls that would be destroyed
            potential_walls = 0
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                for i in range(1, 5):
                    nx, ny = x + dx*i, y + dy*i
                    if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                        if wrld.wall_at(nx, ny):
                            potential_walls += 1
                    else:
                        break  # Stop checking this direction if we hit boundary
            reward += potential_walls * 5  # Reward for potentially destroying walls
            reward -= 2  # Base penalty for bomb placement
        
        # Event-based rewards
        for event in events:
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                reward -= 100
            elif event.tpe == Event.BOMB_HIT_CHARACTER:
                reward -= 200
            elif event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward += 200
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 50
            elif event.tpe == Event.BOMB_HIT_WALL:
                reward += 10
        
        return reward

    def do(self, wrld):
        """Main game loop"""
        # Get current state
        state = self.get_state_features(wrld)
        if state is None:
            return
            
        # Get possible actions
        actions, valid_indices = self.get_possible_actions(wrld, (self.x, self.y))
        if not actions:  # If no valid actions available
            return
            
        # Choose action (ε-greedy)
        if random.random() < self.epsilon:
            # Choose a random action from available actions
            action_pair = random.choice(actions)
            action_idx = self.all_actions.index((action_pair[0][0], action_pair[0][1], action_pair[1]))
        else:
            with torch.no_grad():
                q_values = self.main_network(state)
                # Filter only valid actions
                valid_q = torch.tensor([q_values[i] for i in valid_indices])
                best_valid_idx = valid_q.argmax().item()
                action_idx = valid_indices[best_valid_idx]
                action_pair = actions[valid_indices.index(action_idx)]
        
        # Extract direction and bomb decision
        direction, place_bomb = action_pair
        
        # Execute action
        if place_bomb:
            self.place_bomb()
        self.move(direction[0], direction[1])
        
        # Get new state and reward
        new_wrld, events = wrld.next()
        next_state = self.get_state_features(new_wrld)
        reward = self.get_reward(wrld, (self.x, self.y), (direction, place_bomb), 
                               (self.x + direction[0], self.y + direction[1]), events)
        
        # Store experience
        done = any(e.tpe == Event.CHARACTER_KILLED_BY_MONSTER or 
                  e.tpe == Event.CHARACTER_FOUND_EXIT for e in events)
        
        # Ensure action_idx is within bounds
        action_idx = min(action_idx, self.action_size - 1)
        self.store_experience(state, action_idx, reward, next_state, done)
        
        # Train network
        self.train_network()
        
        # Update target network periodically
        self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Increment step counter
        self.steps += 1

    # Helper methods from testcharacter.py
    def calculate_danger_grid(self, wrld):
        # ... (copy from testcharacter.py)
        pass

    def calculate_bomb_grid(self, wrld):
        # ... (copy from testcharacter.py)
        pass

    def is_valid_move(self, wrld, state, action):
        """Check if a move is valid and safe"""
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        
        # Basic validity checks
        if not (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
            return False
        if wrld.wall_at(nx, ny):
            return False
            
        # Check for immediate monster danger in target position
        if any(wrld.monsters_at(mx, my) 
              for mx, my in self.get_neighbors(wrld, (nx, ny), include_diagonal=True)):
            return False
            
        # Check for explosions at target position
        if wrld.explosion_at(nx, ny):
            return False
            
        # Check for active bombs - only avoid if they're about to explode
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):
                    bomb = wrld.bomb_at(x, y)
                    # Only consider it dangerous if bomb timer is 0 or 1
                    if bomb.timer <= 1 and (nx, ny) in self.get_explosion_positions(wrld):
                        return False
        
        return True

    def get_neighbors(self, wrld, pos, include_diagonal=True):
        """Returns valid neighboring positions"""
        x, y = pos
        neighbors = []
        
        # Define directions (including diagonals if specified)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if include_diagonal:
            directions.extend([(1, 1), (-1, -1), (1, -1), (-1, 1)])
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < wrld.width() and 0 <= ny < wrld.height() 
                and not wrld.wall_at(nx, ny)):
                neighbors.append((nx, ny))
                
        return neighbors

    def heuristic(self, pos, goal):
        """Calculate Manhattan distance between two points"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_explosion_positions(self, wrld):
        """Get all positions that are in range of bombs or current explosions"""
        explosion_positions = set()
        
        # First check for active explosions
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.explosion_at(x, y):
                    explosion_positions.add((x, y))
        
        # Then check for bomb ranges
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):
                    # Add the bomb's position
                    explosion_positions.add((x, y))
                    
                    # Add positions in all four directions (up to 4 cells)
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dx, dy in directions:
                        for i in range(1, 5):  # Range of 4 cells
                            nx, ny = x + (dx * i), y + (dy * i)
                            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                                if wrld.wall_at(nx, ny):
                                    break  # Stop this direction if we hit a wall
                                explosion_positions.add((nx, ny))
                            else:
                                break  # Stop if we're out of bounds
        
        return explosion_positions

    def get_closest_monster_distance(self, wrld, pos):
        """Calculate Manhattan distance to closest monster"""
        x, y = pos
        min_distance = float('inf')
        
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.monsters_at(i, j):
                    distance = abs(x - i) + abs(y - j)
                    min_distance = min(min_distance, distance)
                    
        return min_distance if min_distance != float('inf') else 8  # Return 8 if no monsters found

    def get_safe_neighbors(self, wrld, pos):
        """Get neighboring positions that are safe from monsters and explosions"""
        neighbors = self.get_neighbors(wrld, pos, include_diagonal=True)
        explosion_positions = self.get_explosion_positions(wrld)
        
        safe_neighbors = []
        for nx, ny in neighbors:
            # Check if the position is safe (no monsters adjacent, not in explosion range)
            if not any(wrld.monsters_at(mx, my) 
                      for mx, my in self.get_neighbors(wrld, (nx, ny), include_diagonal=True)):
                if (nx, ny) not in explosion_positions:
                    safe_neighbors.append((nx, ny))
                    
        return safe_neighbors



    def save_model(self):
        """Save the trained model"""
        print(f"Saving model to: {self.model_path}")
        torch.save(self.main_network.state_dict(), self.model_path)

    def load_model(self):
        """Load the trained model if it exists"""
        if os.path.exists(self.model_path):
            print(f"Loading model from: {self.model_path}")
            self.main_network.load_state_dict(torch.load(self.model_path))
            self.target_network.load_state_dict(self.main_network.state_dict())


    def get_path_positions(self, start, goal):
        """Get positions along path to goal"""
        positions = []
        x, y = start
        gx, gy = goal
        while (x, y) != (gx, gy):
            if x < gx:
                x += 1
            elif x > gx:
                x -= 1
            if y < gy:
                y += 1
            elif y > gy:
                y -= 1
            positions.append((x, y))
        return positions

    def is_backtracking_needed(self, wrld, pos, goal):
        """Check if we need to backtrack to avoid monsters"""
        x, y = pos
        # Check if monsters are blocking direct path to goal
        path_positions = self.get_path_positions(pos, goal)
        for px, py in path_positions[:5]:  # Check first 5 positions
            if any(wrld.monsters_at(mx, my) 
                  for mx, my in self.get_neighbors(wrld, (px, py), include_diagonal=True)):
                return True
        return False 