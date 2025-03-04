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
    update_counter = False

    monster_present = False
    explosion_present = False
    bomb_location = None
    danger_grid = None


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
        self.turn_counter += 1
        print(self.turn_counter)

        me = wrld.me(self)  # Get current character state
        state = (me.x, me.y)  # Get character's starting position
        self.danger_grid = self.calculate_danger_grid(wrld) # proximity of each cell to monster
        self.explosion_grid = self.calculate_explosion_grid(wrld)
        self.goal = (wrld.width() - 1, wrld.height() - 1) 

        
        # Choose an action (ε-greedy)
        possible_actions = self.get_possible_actions(wrld, state)
        action = self.choose_action(wrld, state, possible_actions)
   
        direction, place_bomb = action
        if place_bomb:
            self.place_bomb()  # Place the bomb if the action is to place a bomb
            self.turn_counter = 0
        # Move in the chosen direction
        self.move(direction[0], direction[1])

        # Apply action and observe the next state
        next_state = (state[0] + direction[0], state[1] + direction[1])
        reward = self.get_reward(wrld, state, action, next_state)

        # Update Q-learning weights
        self.update_weights(wrld, state, action, reward, next_state)


        print("Updated weights:", self.weights)


    def get_features(self, wrld, state, action):
        x, y = state
        direction, place_bomb = action
        dx, dy = direction
        state_prime = (x + dx, y + dy)

        if not self.is_valid_move(wrld, state, direction):
            return {} 
        
        # Calculate features
        if self.monster_present:
            monster_proximity = -1 * self.get_proximity_cost(wrld, state_prime)
        else:
            monster_proximity = None
        if self.explosion_present:
            explosion_proximity = self.explosion_grid[state_prime]
        else:
            explosion_proximity = None
        if self.turn_counter > 6 and self.turn_counter < 10:
            bomb_proximity = self.get_bomb_cost(wrld, state_prime)
        else:
            bomb_proximity = None

        wall_proximity, edge_proximity = self.get_barrier_proximity(wrld, state_prime)
        cost_to_exit = 1 / self.a_star(wrld, state_prime, self.goal)

        
        features = {
            "cost to exit": cost_to_exit,
            "monster proximity": monster_proximity,
            "wall proximity": wall_proximity,
            "edge proximity": edge_proximity,
            "explosion proximity": explosion_proximity,
            "bomb proximity": bomb_proximity
        }
        
        return features


    def get_q_value(self, wrld, state, action):
        features = self.get_features(wrld, state, action)
        # Filter out features with None values
        valid_features = {f: value for f, value in features.items() if value is not None}
        return sum(self.weights.get(f, 0) * value for f, value in valid_features.items())


    def choose_action(self, wrld, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)  # Exploration
        return max(possible_actions, key=lambda a: self.get_q_value(wrld, state, a))  # Exploitation


    def update_weights(self, wrld, state, action, reward, next_state):
        print("Before update:", self.weights)
        features = self.get_features(wrld, state, action)
        max_next_q = max(self.get_q_value(wrld, next_state, a) for a in self.get_possible_actions(wrld, next_state))
        q_value = self.get_q_value(wrld, state, action)
        delta = (reward + (self.gamma * max_next_q)) - q_value  # Temporal Difference error

        for feature, value in features.items():
            if feature not in self.weights:
                self.weights[feature] = 0
            if value is not None:
                self.weights[feature] += self.alpha * delta * value

                # Clip weights to prevent overflow
                self.weights[feature] = max(min(self.weights[feature], 500), -500)

        self.save_weights()  # Save updated weights to file


    def get_reward(self, wrld, state, action, next_state):
        if wrld.monsters_at(*next_state):
            reward = -1000  # Heavy penalty for getting caught
        elif wrld.bomb_at(*next_state):
            reward = -1000  # Avoid bombs
        elif wrld.explosion_at(*next_state):
            reward = -1000  
        elif wrld.exit_at(*next_state):
            reward = 100  # High reward for reaching the exit
        else:
            reward = -1  # Small penalty for normal movement
        return reward
        

    def get_possible_actions(self, wrld, state):
        # Define direction actions (up, down, left, right, and diagonals)
        directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # Initialize actions list with direction and 'place_bomb' boolean
        actions = []
        
        for direction in directions:
            # Add action where placing a bomb is an option (True/False)
            actions.append((direction, False))  # First, without placing a bomb
            actions.append((direction, True))   # Second, with placing a bomb
            
        # Filter the actions based on whether the move is valid or not
        actions = [(direction, place_bomb) for direction, place_bomb in actions if self.is_valid_move(wrld, state, direction)]
        print(actions)

        return actions



    def is_valid_move(self, wrld, state, action):
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        return 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx,ny)


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
                total_cost = f_score[current] + 1
                return total_cost
            
            for neighbor in self.get_neighbors(wrld, current):
                temp_g_score = g_score[current] + 1 #+ self.get_proximity_cost(neighbor)
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return 10000 # No path found

    def calculate_danger_grid(self, wrld):
        """Calculates the proximity of monsters for all cells in the world."""
        self.monster_present = False
        # Initialize the danger grid with a high value
        danger_grid = { (x, y): float('inf') for x in range(wrld.width()) for y in range(wrld.height()) }

        # Initialize BFS queue and set proximity for monster positions
        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.monsters_at(x, y):  # If there is a monster in the cell
                    danger_grid[(x, y)] = 0  # The monster cell itself has proximity 0
                    queue.append((x, y))  # Add to BFS queue
                    self.monster_present = True
        
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
    

    def calculate_explosion_grid(self, wrld):
        self.explosion_present = False
        explosion_grid = { (x, y): 0 for x in range(wrld.width()) for y in range(wrld.height()) }

        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.explosion_at(x, y):  
                    explosion_grid[(x, y)] = 1 
                    queue.append((x, y)) 
                    self.explosion_present = True

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # BFS to propagate danger levels
        while queue:
            x, y = queue.popleft()
            
            # If we're already at distance 3, stop expanding further
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and explosion_grid[(nx, ny)] == 0:
                    explosion_grid[(nx, ny)] = .5
        
        return explosion_grid
    
    def calculate_bomb_grid(self, wrld):
        self.explosion_present = False
        explosion_grid = { (x, y): 0 for x in range(wrld.width()) for y in range(wrld.height()) }

        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.explosion_at(x, y):  
                    explosion_grid[(x, y)] = 1 
                    queue.append((x, y)) 
                    self.explosion_present = True

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # BFS to propagate danger levels
        while queue:
            x, y = queue.popleft()
            
            # If we're already at distance 3, stop expanding further
            for dx, dy in directions:
                nx, ny = x, y
                for i in range(6):
                    nx += dx
                    ny += dy
                    if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and explosion_grid[(nx, ny)] == 0:
                        explosion_grid[(nx, ny)] = 1
        
        return explosion_grid
        
    
    def get_proximity_cost(self, wrld, state):
        """Returns the proximity cost for a given cell (x, y)."""
        if not self.is_valid_move(wrld, state, (0,0)):
            return 500
        dist = self.danger_grid[state]
        
        if dist == 0:  # Monster cell itself (wall)
            return 40
        elif dist == 1:  # Cells around the monster (immediate danger)
            return 30  # Wall-like behavior
        elif dist == 2:  # Two cells away from monster (heavy penalty)
            return 20  # Heavy penalty
        elif dist == 3:  # Three cells away from monster (light penalty)
            return 10  # Light penalty
        elif dist == 4:  # Four cells away from monster (slight penalty)
            return 5  # Light penalty
        return 1  # Default cost for free space
    
        
    def get_bomb_cost(self, wrld, state):
        bomb_grid  = self.calculate_bomb_grid(wrld)
        """Returns the proximity cost for a given cell (x, y)."""
        if not self.is_valid_move(wrld, state, (0,0)) or bomb_grid[state] == 1:
            return 500
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
    
    def get_barrier_proximity(self, wrld, state):
        wall_score = -1
        edge_score = -1
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
                    # edge penalty
                    edge_score -= 1

        return wall_score, edge_score