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
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from datetime import datetime

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
        
        # Set absolute path for model save
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, "trained_model.pth")
        
        # Print full path and check for access
        print(f"Model path: {os.path.abspath(self.model_path)}")
        print(f"Directory exists: {os.path.exists(os.path.dirname(self.model_path))}")
        print(f"Directory writable: {os.access(os.path.dirname(self.model_path), os.W_OK)}")
        
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
        self.epsilon_decay = 0.9999816  # Decay rate for epsilon for 250,000 steps (5000 steps per episode and 50 episodes)
        self.learning_rate = 0.0005
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
        self.update_target_every = 50  # Update target network every 50 steps
        self.total_reward = 0.0  # Track total accumulated reward

        # Attempt to load model right away
        loaded = self.load_model()
        if loaded:
            print("✅ Successfully loaded existing model")
        else:
            print("⚠️ Starting with a new model")
        
        # Add autosave feature
        self.last_save_time = datetime.now()
        self.save_interval_seconds = 30  # Save every 30 seconds

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


    """ def get_reward(self, wrld, state, action, next_state, events):
        #Calculate reward with stronger focus on vertical movement
        reward = 0
        x, y = next_state
        
        # Goal is always the bottom-right corner
        goal = (wrld.width() - 1, wrld.height() - 1)
        
        # Base reward based on distance to goal
        current_distance = self.heuristic(state, goal)
        next_distance = self.heuristic(next_state, goal)
        reward += (current_distance - next_distance) * 2
        
        # SPECIAL REWARD FOR VERTICAL PROGRESS
        # Give extra reward for moving down (increasing y)
        if next_state[1] > state[1]:
            reward += 3.0  # Big bonus for moving down
        elif next_state[1] < state[1]:
            reward -= 2.0  # Penalty for moving up
        
        # Check if path to exit is clear
        path_positions = self.get_path_positions((x, y), goal)
        walls_in_path = sum(1 for px, py in path_positions if wrld.wall_at(px, py))
        
        # If path is clear, strongly reward moving toward exit
        if walls_in_path == 0:
            if next_distance < current_distance:
                reward += 5.0
            else:
                reward -= 5.0
        
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
        
        return reward """
    
    def get_reward(self, wrld, state, action, next_state, events):
        reward = 0
        
        # Goal is the exit
        goal = (wrld.width() - 1, wrld.height() - 1)
        
        # Reward progress toward goal more uniformly
        current_distance = self.heuristic(state, goal)
        next_distance = self.heuristic(next_state, goal)
        reward += (current_distance - next_distance) * 3
        
        # Simple event-based rewards
        for event in events:
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                reward -= 100
            elif event.tpe == Event.BOMB_HIT_CHARACTER:
                reward -= 100
            elif event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward += 200
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 30
            elif event.tpe == Event.BOMB_HIT_WALL:
                reward += 5
        
        return reward

    def do(self, wrld):
        """Main game loop with time-based autosave"""
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
        
        # Save model on important events
        if any(e.tpe == Event.CHARACTER_FOUND_EXIT for e in events):
            print(f"🎯 Exit found! Saving model... Total reward: {self.total_reward:.2f}")
            self.save_model()
            self.last_save_time = datetime.now()
        elif any(e.tpe == Event.BOMB_HIT_WALL for e in events):
            print("💥 Wall destroyed! Saving model...")
            self.save_model()
            self.last_save_time = datetime.now()


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
        """Save the trained model with robust error handling"""
        try:
            print(f"💾 Attempting to save model to: {self.model_path}")
            
            # Save all necessary information
            state = {
                'main_network': self.main_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps,
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
                self.steps = checkpoint['steps']
                
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

# Add this at the end of the file to force a save when the module is unloaded
import atexit
import json

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
                for event in reversed(testcharacter_instance.last_events):
                    if event.tpe == Event.CHARACTER_FOUND_EXIT:
                        outcome = "win"
                        break
                    elif event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                        outcome = "killed_by_monster"
                        break
                    elif event.tpe == Event.BOMB_HIT_CHARACTER:
                        outcome = "killed_by_bomb"
                        break
            
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