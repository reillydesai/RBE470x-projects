# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import heapq
import math
from collections import deque
import random

import json
import os

class TestCharacter(CharacterEntity):

    turn_counter = 0
    bomb1_dropped = False
    bomb2_dropped = False
    chased = False
    bomb_location = None
    danger_grid = None
    goal = (0,0)

    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    weights = {}  # Feature weights
    min_cost_to_exit = 0
    max_cost_to_exit = 0

    def __init__(self, name, avatar, x=0, y=0, *args, **kwargs):
        # If avatar is not needed, you can pass a placeholder like `None`
        super().__init__(name, avatar, x, y)
        self.weights_file = "weights.json"
        self.weights = self.load_weights()
        
    

    def do(self, wrld):

        me = wrld.me(self)  # Get current character state
        state= (me.x, me.y)  # Get character's starting position
        self.danger_grid = self.calculate_danger_grid(wrld) # proximity of each cell to monster
        self.goal = (wrld.width() - 1, wrld.height() - 1) 

        
        # Choose an action (ε-greedy)
        possible_actions = self.get_possible_actions(wrld, state)
        action = self.choose_action(wrld, state, possible_actions)

        # Apply action and observe the next state
        next_state = (state[0] + action[0], state[1] + action[1])
        reward = self.get_reward(wrld, state, action, next_state)

        # Update Q-learning weights
        self.update_weights(wrld, state, action, reward, next_state)

        # Execute the action
        self.move(action[0], action[1])  # Move in chosen direction

        print("Updated weights:", self.weights)



    def get_features(self, wrld, state, action):
        x, y = state
        dx, dy = action
        state_prime = (x + dx, y + dy)
        
        # Calculate features\
        monster_proximity = self.get_proximity_cost(state_prime)
        wall_proximity = self.get_wall_proximity(wrld, state_prime)
        cost_to_exit = self.a_star(wrld, state_prime, self.goal)
        
        # Update min and max cost_to_exit values dynamically
        self.min_cost_to_exit = min(self.min_cost_to_exit, cost_to_exit)
        self.max_cost_to_exit = max(self.max_cost_to_exit, cost_to_exit)
        
        # Normalize cost_to_exit dynamically based on observed min/max
        if self.max_cost_to_exit != self.min_cost_to_exit:  # Avoid division by zero
            normalized_cost_to_exit = (cost_to_exit - self.min_cost_to_exit) / (self.max_cost_to_exit - self.min_cost_to_exit)
        else:
            normalized_cost_to_exit = 0 
        
        
        # Normalize monster proximity to [0, 1] (assuming max value is 2000)
        normalized_monster_proximity = min(max(monster_proximity, 0), 2000) / 2000
        
        # Normalize wall proximity (assuming the wall score is in [-5, 5])
        normalized_wall_proximity = (wall_proximity + 5) / 10  # Now between 0 and 1
        
        features = {
            "cost to exit": normalized_cost_to_exit,
            "monster proximity": normalized_monster_proximity,
            "wall proximity": normalized_wall_proximity
        }
        
        return features


    def get_q_value(self, wrld, state, action):
        features = self.get_features(wrld, state, action)
        return sum(self.weights.get(f, 0) * value for f, value in features.items())

    def choose_action(self, wrld, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)  # Exploration
        return max(possible_actions, key=lambda a: self.get_q_value(wrld, state, a))  # Exploitation

    # def update_weights(self, wrld, state, action, reward, next_state):
    #     print("Before update:", self.weights)
    #     features = self.get_features(wrld, state, action)
    #     max_next_q = max(self.get_q_value(wrld, next_state, a) for a in self.get_possible_actions(wrld, next_state))
    #     q_value = self.get_q_value(wrld, state, action)
    #     td_error = reward + self.gamma * max_next_q - q_value  # Temporal Difference error

    #     for feature, value in features.items():
    #         if feature not in self.weights:
    #             self.weights[feature] = 0  # Initialize weight if not set
    #         self.weights[feature] += self.alpha * td_error * value  # Update weights

    #     self.save_weights()  # Save updated weights to file


    def update_weights(self, wrld, state, action, reward, next_state):
        print("Before update:", self.weights)
        features = self.get_features(wrld, state, action)
        max_next_q = max(self.get_q_value(wrld, next_state, a) for a in self.get_possible_actions(wrld, next_state))
        q_value = self.get_q_value(wrld, state, action)
        td_error = reward + self.gamma * max_next_q - q_value  # Temporal Difference error

        # Apply a cap to the temporal difference error (to avoid large updates)
        td_error = max(min(td_error, 10), -10)

        for feature, value in features.items():
            if feature not in self.weights:
                self.weights[feature] = 0
            self.weights[feature] += self.alpha * td_error * value

            # Clip weights to prevent overflow
            self.weights[feature] = max(min(self.weights[feature], 50), -50)

        self.save_weights()  # Save updated weights to file


    def get_reward(self, wrld, state, action, next_state):
    #     # Standard reward structure
    #     if wrld.monsters_at(*next_state):
    #         reward = -1000  # Heavy penalty for getting caught
    #     elif wrld.exit_at(*next_state):
    #         reward = 1000  # Huge reward for reaching exit
    #     elif wrld.bomb_at(*next_state):
    #         reward = -500  # Avoid bombs
    #     else:
    #         reward = -1  # Small penalty to encourage movement
        
    #     # Normalize the reward to [-1, 1]
    #     if reward > 1000:
    #         reward = 1
    #     elif reward < -1000:
    #         reward = -1
    #     else:
    #         reward = reward / 1000  # Scale to [-1, 1]
        
    #     return reward

        if wrld.exit_at(*next_state):
            reward = 10  # High reward for reaching the exit
        else:
            reward = -1  # Small penalty for normal movement
        return reward


        
        

    def get_possible_actions(self, wrld, state):
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]  
        return [a for a in actions if self.is_valid_move(wrld, state, a)]

    def is_valid_move(self, wrld, state, action):
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        return 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny)


    def save_weights(self):
        if not self.weights_file:
            print("Error: weights_file is not set!")
            return
        
        try:
            with open(self.weights_file, "w") as f:
                json.dump(self.weights, f)
            print(f"Weights successfully saved to {self.weights_file}")
        except Exception as e:
            print(f"Error saving weights: {e}")


    def load_weights(self):
        if self.weights_file is None:
            print("Error: weights_file is None!")
            return {}
        
        if os.path.exists(self.weights_file):
            with open(self.weights_file, "r") as f:
                self.weights = json.load(f)
        else:
            self.weights = {}
        
        return self.weights






    def a_star(self, wrld, start, goal):
        """Finds the shortest path from start to goal using A*."""
        open_set = []  # Priority queue
        heapq.heappush(open_set, (0, start))  # (f-score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                total_cost = f_score[current]
                return total_cost
            
            for neighbor in self.get_neighbors(wrld, current):
                temp_g_score = g_score[current] + 1 + self.get_proximity_cost(neighbor)
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, float('inf') # No path found

    def calculate_danger_grid(self, wrld):
        """Calculates the proximity of monsters for all cells in the world."""
        # Initialize the danger grid with a high value
        danger_grid = { (x, y): float('inf') for x in range(wrld.width()) for y in range(wrld.height()) }

        # Initialize BFS queue and set proximity for monster positions
        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.monsters_at(x, y):  # If there is a monster in the cell
                    danger_grid[(x, y)] = 0  # The monster cell itself has proximity 0
                    queue.append((x, y))  # Add to BFS queue
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # BFS to propagate danger levels
        while queue:
            x, y = queue.popleft()
            
            # If we're already at distance 3, stop expanding further
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and danger_grid[(nx, ny)] == float('inf'):
                    danger_grid[(nx, ny)] = danger_grid[(x, y)] + 1
                    queue.append((nx, ny))
                    if danger_grid[(nx, ny)] == 4:  # Stop expanding beyond 4 cells
                        continue
        
        return danger_grid

    def get_proximity_cost(self, state):
        """Returns the proximity cost for a given cell (x, y)."""
        dist = self.danger_grid[state]
        
        if dist == 0:  # Monster cell itself (wall)
            return 20
        elif dist == 1:  # Cells around the monster (immediate danger)
            return 10  # Wall-like behavior
        elif dist == 2:  # Two cells away from monster (heavy penalty)
            return 5  # Heavy penalty
        elif dist == 3:  # Three cells away from monster (light penalty)
            return 3  # Light penalty
        elif dist == 4:  # Four cells away from monster (slight penalty)
            return 2  # Light penalty
        return 1  # Default cost for free space
    

    def heuristic(self, pos, goal):
        """Euclidean distance heuristic."""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def get_neighbors(self, wrld, pos):
        """Returns valid neighboring positions (up, down, left, right) ignoring walls."""
        x, y = pos
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny):
                neighbors.append((nx, ny))

        return neighbors
    
    def get_wall_proximity(self, wrld, state):
        wall_score = 1
        x, y = state
        # Check surrounding cells within a 2-cell radius and avoid edges
        for dx in range(-2, 3):  # From -2 to 2
            for dy in range(-2, 3):  # From -2 to 2
                nx, ny = x + dx, y + dy
                # Ensure the new cell is within bounds
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    # If a wall is found at the cell
                    if wrld.wall_at(nx, ny):
                        wall_score -= 1
                else:
                    # If the cell is out of bounds, treat it as a wall
                    wall_score -= 1

        return wall_score