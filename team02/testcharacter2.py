# Import section
import sys
sys.path.insert(0, '../bomberman')  # Adds the bomberman directory to Python's path for imports

# Import game-related classes
from entity import CharacterEntity  # Base class for characters
from events import Event  # Event handling

# Neural network related imports
import torch  # Main PyTorch library
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torch.nn.functional as F  # Neural network functions
import numpy as np  # Numerical operations
from collections import deque  # For experience replay buffer
import random  # For random actions
import os  # File operations
from datetime import datetime  # Time tracking
import atexit  # Register exit handler
import json  # JSON handling for logging
import heapq  # Add this with other imports

# Define the neural network architecture
class DQNetwork(nn.Module):
    """Neural network for Deep Q-learning"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        # Sequential network with 2 hidden layers
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input layer → Hidden layer 1
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_size, hidden_size),  # Hidden layer 1 → Hidden layer 2
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_size, output_size)  # Hidden layer 2 → Output layer
        )
    
    def forward(self, x):
        return self.network(x)  # Process input through network

# Main character class
class TestCharacter(CharacterEntity):
    def __init__(self, name, avatar, x=0, y=0):
        super().__init__(name, avatar, x, y)  # Initialize parent class
        # Set up model saving path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, "trained_model.pth")
        
        # Debug prints for file access
        print(f"Model path: {os.path.abspath(self.model_path)}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(self.model_path))}")
        print(f"Directory writable: {os.access(os.path.dirname(self.model_path), os.W_OK)}")
        
        # Define all possible actions as (dx, dy, place_bomb)
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
        
        # DQL Hyperparameters
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Starting exploration rate (100% random actions)
        self.epsilon_min = 0.01  # Minimum exploration rate (1% random actions)
        self.epsilon_decay = 0.9999816  # How fast epsilon decreases
        self.learning_rate = 0.0005  # Learning rate for optimizer
        self.batch_size = 32  # Number of experiences to learn from at once
        
        # Network dimensions
        self.state_size = 10  # Input features size
        self.action_size = 20  # Number of possible actions
        
        # Create neural networks
        self.main_network = DQNetwork(self.state_size, 64, self.action_size)  # Main network
        self.target_network = DQNetwork(self.state_size, 64, self.action_size)  # Target network
        self.target_network.load_state_dict(self.main_network.state_dict())  # Copy weights
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)  # Store last 10000 experiences
        
        # Training tracking variables
        self.steps = 0  # Count of training steps
        self.update_target_every = 50  # Update target network every 50 steps
        self.total_reward = 0.0  # Track cumulative reward
        
        # Try to load existing model
        loaded = self.load_model()
        if loaded:
            print("✅ Successfully loaded existing model")
        else:
            print("⚠️ Starting with a new model")
        
        # Set up autosave
        self.last_save_time = datetime.now()
        self.save_interval_seconds = 30  # Save every 30 seconds


    def do(self, wrld):
        """Main game loop with A* override"""

        print("steps: ",self.steps)
        # Check for clear path to exit
        clear_path_exists, path = self.is_clear_path_to_exit(wrld)
        
        if clear_path_exists and path:
            # Use A* path
            next_pos = path[0]
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            
            # Store current state for reward calculation
            current_state = (self.x, self.y)
            
            # Don't place bombs when we have a clear path
            self.move(dx, dy)
            
            # Get events after movement
            new_wrld, events = wrld.next()
            self.last_events = events
            
            # Calculate reward including win condition
            reward = 100  # Base reward for following optimal path
            
            # Use the regular reward function to ensure we get win rewards
            reward += self.get_reward(wrld, current_state, ((dx, dy), False), 
                                    (self.x, self.y), events)
            
            # Update total reward
            self.total_reward += reward
            
            # Increment step counter
            self.steps += 1
            
            return
            
        # If no clear path, continue with normal DQN behavior
        # Initialize events as empty list by default
        events = []
        
        # Auto-save model based on time
        current_time = datetime.now()
        time_diff = (current_time - self.last_save_time).total_seconds()
        
        if time_diff >= self.save_interval_seconds:
            print(f"⏰ Auto-saving model ({time_diff:.1f} seconds since last save)")
            self.save_model()
            self.last_save_time = current_time
        
        # Get current state
        state = self.get_state_features(wrld)
        if state is None:
            self.last_events = events  # Store empty events list
            return
            
        # Get possible actions
        actions, valid_indices = self.get_possible_actions(wrld, (self.x, self.y))
        if not actions:  # If no valid actions available
            self.last_events = events  # Store empty events list
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
        
        # Store the events for logging with debug print
        self.last_events = events
        print(f"🎮 Current events: {[str(e.tpe) for e in events]}")  # Debug print
        
        next_state = self.get_state_features(new_wrld)
        reward = self.get_reward(wrld, (self.x, self.y), (direction, place_bomb), 
                               (self.x + direction[0], self.y + direction[1]), events)
        
        # Update total reward
        self.total_reward += reward
        
        # Store experience
        done = any(e.tpe == Event.CHARACTER_KILLED_BY_MONSTER or Event.BOMB_HIT_CHARACTER
                  or e.tpe == Event.CHARACTER_FOUND_EXIT for e in events)
        
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
        
        # Save model on important events
        if any(e.tpe == Event.CHARACTER_FOUND_EXIT for e in events):
            print(f"🎯 Exit found! Saving model... Total reward: {self.total_reward:.2f}")
            self.save_model()
            self.last_save_time = datetime.now()
        elif any(e.tpe == Event.CHARACTER_KILLED_BY_MONSTER for e in events):
            print("💥 Monster killed! Saving model...")
            self.save_model()
            self.last_save_time = datetime.now()
        elif any(e.tpe == Event.BOMB_HIT_CHARACTER for e in events):
            print("💥 Bomb killed! Saving model...")
            self.save_model()
            self.last_save_time = datetime.now()

## ACTIONS

    def get_possible_actions(self, wrld, state):
        """Get list of possible actions and their indices"""
        actions = []
        valid_indices = []
        
        for i, (dx, dy, place_bomb) in enumerate(self.all_actions):
            if i < self.action_size and self.is_valid_move(wrld, state, (dx, dy)):
                actions.append(((dx, dy), place_bomb))
                valid_indices.append(i)
                    
        return actions, valid_indices
    
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


### REWARDS

    def get_reward(self, wrld, state, action, next_state, events):
        reward = 0

        (dx, dy), place_bomb = action
        
        # Goal is the exit
        goal = (wrld.width() - 1, wrld.height() - 1)
        
        # Reward progress toward goal more uniformly
        current_distance = self.heuristic(state, goal)
        next_distance = self.heuristic(next_state, goal)
        reward += (current_distance - next_distance) * 40
        
        # Time penalty (small negative reward each step)
        reward -= 10 # Penalize taking time
        
        # Simple event-based rewards
        for event in events:
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                reward -= 800  # Increased death penalty
            elif event.tpe == Event.BOMB_HIT_CHARACTER:
                reward -= 800  # Increased death penalty
            elif event.tpe == Event.CHARACTER_FOUND_EXIT:
                # Base reward for winning
                reward += 15000
                
                # Bonus reward based on steps taken (more steps = less bonus)
                time_bonus = max(500 - self.steps, 0)  # Starts at 500, decreases with steps
                reward += time_bonus
                
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 100  # Increased monster kill reward
            elif event.tpe == Event.BOMB_HIT_WALL:
                # Only reward wall destruction if it's between us and the goal
                if self.is_wall_blocking_path(wrld, state, goal):
                    reward += 300
                else:
                    reward += 10

            if place_bomb:
                reward -= 100
        
        return reward

    def is_wall_blocking_path(self, wrld, state, goal):
        """Check if destroying this wall is the first one in a complete horizontal wall"""
        x, y = state
        
        # Count walls in this row
        wall_count = 0
        for check_x in range(wrld.width()):
            if wrld.wall_at(check_x, y):
                wall_count += 1
                
        # If this was the first wall destroyed in a complete row
        # (wall_count + 1 because the wall we just destroyed isn't counted)
        if wall_count + 1 == wrld.width():
            return True
            
        return False

### FEATURES AND HELPER FUNCTIONS

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
        """
        Prioritize y-coordinate movement (vertical) over x-coordinate movement
        This helps the agent focus on getting past walls to reach the exit
        """
        x1, y1 = pos
        x2, y2 = goal
        
        # Heavily weight y-distance (3x) compared to x-distance
        y_weight = 3.0
        x_weight = 1.0
        
        # Calculate weighted Manhattan distance
        return x_weight * abs(x2 - x1) + y_weight * abs(y2 - y1)

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
                    
                    # Add positions in all four directions (up to 6 cells)
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dx, dy in directions:
                        for i in range(1, 7):  # Range of 6 cells 
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
    

### TRAINING & LOGGING HELPERS

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


        #Implement double DQN
        with torch.no_grad():
            # Main network selects actions
            next_action_indices = self.main_network(next_states).argmax(dim=1, keepdim=True)
            # Target network evaluates those actions
            next_q_values = self.target_network(next_states).gather(1, next_action_indices).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

    def update_target_network(self):
        """Update target network weights"""
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

    def save_model(self):
        """Save the trained model with robust error handling"""
        try:
            print(f"💾 Attempting to save model to: {self.model_path}")
            
            # Save all necessary information
            state = {
                'main_network': self.main_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save to temporary file first
            temp_path = self.model_path + ".tmp"
            torch.save(state, temp_path)
            
            # If successful, rename to actual file
            if os.path.exists(temp_path):
                if os.path.exists(self.model_path):
                    os.remove(self.model_path)  # Remove existing file
                os.rename(temp_path, self.model_path)
                print(f"✅ Model successfully saved to {self.model_path}")
                return True
            else:
                print("❌ Error: Temporary file not created")
                return False
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            
            # Try alternative locations
            try:
                alt_paths = [
                    os.path.join(os.getcwd(), "trained_model.pth"),
                    "/tmp/bomberman_model.pth",
                    os.path.expanduser("~/bomberman_model.pth")
                ]
                
                for alt_path in alt_paths:
                    try:
                        print(f"📁 Trying alternative location: {alt_path}")
                        torch.save(state, alt_path)
                        print(f"✅ Model saved to alternative location: {alt_path}")
                        return True
                    except Exception as e2:
                        print(f"❌ Failed to save to {alt_path}: {e2}")
                        continue
            except Exception as e3:
                print(f"❌ Failed all save attempts: {e3}")
            
            return False

    def load_model(self):
        """Load the trained model with robust error handling"""
        try:
            if os.path.exists(self.model_path):
                print(f"📂 Loading model from: {self.model_path}")
                checkpoint = torch.load(self.model_path)
                
                # Load all saved information
                self.main_network.load_state_dict(checkpoint['main_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint['epsilon']
                
                print(f"✅ Model loaded successfully (saved on {checkpoint.get('timestamp', 'unknown date')})")
                print(f"   → Epsilon: {self.epsilon:.4f}, Steps: {self.steps}")
                return True
            else:
                print(f"⚠️ No model found at {self.model_path}")
                
                # Try alternative locations
                alt_paths = [
                    os.path.join(os.getcwd(), "trained_model.pth"),
                    "/tmp/bomberman_model.pth",
                    os.path.expanduser("~/bomberman_model.pth")
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"📁 Found model at alternative location: {alt_path}")
                        checkpoint = torch.load(alt_path)
                        self.main_network.load_state_dict(checkpoint['main_network'])
                        self.target_network.load_state_dict(checkpoint['target_network'])
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        self.epsilon = checkpoint['epsilon'] 
                        self.steps = checkpoint['steps']
                        print(f"✅ Model loaded from {alt_path}")
                        return True
                
                print("⚠️ No model found in any location")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Starting with a new model")
            return False

    def is_clear_path_to_exit(self, wrld):
        """Check if there's a clear path to the exit using A*"""
        goal = (wrld.width() - 1, wrld.height() - 1)
        start = (self.x, self.y)
        
        # A* implementation
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < wrld.width() and 0 <= ny < wrld.height() and
                    not wrld.wall_at(nx, ny) and
                    not any(wrld.monsters_at(mx, my) 
                           for mx, my in self.get_neighbors(wrld, (nx, ny), include_diagonal=True))):
                    neighbors.append((nx, ny))
            return neighbors
        
        # A* algorithm
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return True, path
            
            for next_pos in get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return False, []


def exit_handler():
    """Save the model and log training metrics when the program exits"""
    print("🔚 Program exiting, saving model...")
    try:
        if 'testcharacter_instance' in globals() and testcharacter_instance is not None:
            # Save the model
            testcharacter_instance.save_model()
            
            # Log training metrics in JSONL format
            log_path = os.path.join(os.path.dirname(testcharacter_instance.model_path), "training_log.jsonl")
            
            # Determine outcome from last event
            outcome = "unknown"
            if hasattr(testcharacter_instance, 'last_events'):
                print(f"Final events: {testcharacter_instance.last_events}")  # Debug print
                
                for event in testcharacter_instance.last_events:
                    event_str = str(event.tpe)

                    # Also check string representations as fallback
                    if "found" in str(event).lower():
                        outcome = "win"
                        break
                    elif "killed by monster" in str(event).lower():
                        outcome = "killed_by_monster"
                        break
                    elif "bomb hit character" in str(event).lower():
                        outcome = "killed_by_bomb"
                        break
                    elif "game over" in str(event).lower() or "time" in str(event).lower():
                        outcome = "timeout"
                        break
                
                if outcome == "unknown":
                    print("⚠️ Warning: Could not determine game outcome from events:", 
                          [str(e.tpe) for e in testcharacter_instance.last_events])
            else:
                print("⚠️ Warning: No last_events found")
            
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epsilon": float(testcharacter_instance.epsilon),
                "steps": int(testcharacter_instance.steps),
                "total_reward": float(testcharacter_instance.total_reward),
                "outcome": outcome
            }
            
            # Append the JSON line to the log file
            with open(log_path, "a") as log_file:
                json.dump(log_entry, log_file)
                log_file.write('\n')
                
            print(f"📝 Training metrics logged to {log_path}")
            print(f"   → Total reward: {testcharacter_instance.total_reward:.2f}")
    except Exception as e:
        print(f"❌ Error in exit handler: {e}")

atexit.register(exit_handler)

# This will be set by the first instance created
testcharacter_instance = None

# Override the constructor to track the instance
original_init = TestCharacter.__init__
def new_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    global testcharacter_instance
    testcharacter_instance = self
    
TestCharacter.__init__ = new_init