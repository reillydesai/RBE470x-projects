# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from sensed_world import SensedWorld
from events import Event

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
    explosion_positions = []
    monster_positions = []
    danger_grid = None
    obstacles = []


    alpha = 0.01 # Learning rate
    gamma = .7 # Discount factor
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
        # self.turn_counter += 1
        # print(self.turn_counter)

        me = wrld.me(self)  # Get current character state
        state = (me.x, me.y)  # Get character's starting position
        new_wrld, events = wrld.next()
        
        self.explosion_positions = self.get_explosion_positions(new_wrld)
        self.danger_grid = self.calculate_danger_grid(wrld)

        self.goal = (wrld.width() - 1, wrld.height() - 1) 

        
        # Choose an action (ε-greedy)
        possible_actions = self.get_possible_actions(wrld, state)
        action = self.choose_action(wrld, state, possible_actions)
   
        direction, place_bomb = action
        if place_bomb:
            self.place_bomb()  
            #self.turn_counter = 0
        self.move(direction[0], direction[1])

        next_state = (state[0] + direction[0], state[1] + direction[1])
        reward = self.get_reward(wrld, state, action, next_state)

        self.update_weights(wrld, state, action, reward, next_state)

        print("Updated weights:", self.weights)


    def get_features(self, wrld, state, action):
        x, y = state
        direction, place_bomb = action
        dx, dy = direction
        state_prime = (x + dx, y + dy)
        new_world, events = wrld.next()

        if not self.is_valid_move(wrld, state, direction):
            return {}  # Invalid move, return no features

        # **New Feature Definitions**
        escape_value = self.escape_from_danger(state_prime, wrld)

        bomb_proximity = self.get_bomb_cost(new_world, state_prime)
        monster_proximity = self.get_proximity_cost(wrld, state)  # Higher = better
        cost_to_exit = 1 / (self.heuristic(state, self.goal) + 1)  # Normalize, avoid div by zero
        safe_tiles = sum(1 for neighbor in self.get_neighbors(wrld, state_prime) if neighbor not in self.obstacles)
        wall_proximity, edge_proximity = self.get_barrier_proximity(wrld, state)

        features = {
            "escape feasibility": escape_value,
            "bomb proximity": bomb_proximity,
            "monster proximity": monster_proximity,
            "exit proximity": cost_to_exit,
            "safe tile availability": safe_tiles,
            "wall proximity": wall_proximity,
            "edge proximity": edge_proximity

        }

        print(features)

        return features

    
    def escape_from_danger(self, state, wrld):

        # Ensure explosion_positions and monster_positions are always lists (empty if None)
        self.explosion_positions = self.explosion_positions or []
        self.monster_positions = self.monster_positions or []

        # Create obstacles set from explosion and monster positions
        self.obstacles = set(self.explosion_positions) | set(self.monster_positions)

        # Check if there are no obstacles
        if self.obstacles == []:
            return 1.0  # No danger, completely safe

        me_x, me_y = state
        safe_tiles = [(x, y) for x in range(me_x-1, me_x+1) for y in range(me_y-1, me_y+1)
                    if (x, y) not in self.obstacles and self.is_valid_move(wrld, state, (x,y))]  # All non-dangerous tiles

        if not safe_tiles:
            return 0  # No safe zone, worst-case scenario

        # Find the shortest path to any safe tile
        min_escape_length = min(self.a_star(wrld, state, safe_tile) for safe_tile in safe_tiles)

        return 1.0 / (1 + min_escape_length)  # Normalize: Closer to 1 means safer, closer to 0 means trapped.



    def get_q_value(self, wrld, state, action):
        features = self.get_features(wrld, state, action)
        # Filter out features with None values
        valid_features = {f: value for f, value in features.items() if value is not None}
        return sum(self.weights.get(f, 0) * value for f, value in valid_features.items())


    def choose_action(self, wrld, state, possible_actions):
        
        valid_actions = [a for a in possible_actions if self.is_valid_move(wrld, state, a[0])]
    
        if random.random() < self.epsilon:
            return random.choice(valid_actions)   # Exploration
        return max(possible_actions, key=lambda a: self.get_q_value(wrld, state, a))  # Exploitation


    def update_weights(self, wrld, state, action, reward, next_state):
        print("Before update:", self.weights)
        features = self.get_features(wrld, state, action)
        max_next_q = max(self.get_q_value(wrld, next_state, a) for a in self.get_possible_actions(wrld, next_state))
        print(max_next_q)
        q_value = self.get_q_value(wrld, state, action)
        delta = (reward + (self.gamma * max_next_q)) - q_value  # Temporal Difference error

        for feature, value in features.items():
            if feature not in self.weights:
                self.weights[feature] = 0
            if value is not None:
                self.weights[feature] += self.alpha * delta * value

                # Clip weights to prevent overflow
                self.weights[feature] = max(min(self.weights[feature], 1000), -1000)

        self.save_weights()  # Save updated weights to file


    def get_reward(self, wrld, state, action, next_state):
        new_world, events = wrld.next()
        x, y = next_state
        
        reward = 0.5 * y * y 
        if self.get_bomb_cost(new_world, next_state) > 1:
            reward += 20

        for event in events:
            if event.tpe == Event.CHARACTER_KILLED_BY_MONSTER:
                reward = -10  # Heavy penalty for getting caught
            elif event.tpe == Event.BOMB_HIT_CHARACTER:
                reward = -20  # Avoid getting hit by bombs
            elif event.tpe == Event.BOMB_HIT_WALL:
                reward += 5  # Reward for destroying walls
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 5  # High reward for killing a monster
            elif event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward = 10  # High reward for reaching the exit

        return reward

        

    def get_possible_actions(self, wrld, state):
        # Define direction actions (up, down, left, right, and diagonals)
        directions = [(0,0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # Initialize actions list with direction and 'place_bomb' boolean
        actions = []
        
        for direction in directions:
            # Add action where placing a bomb is an option (True/False)
            actions.append((direction, False))  # First, without placing a bomb
            #actions.append((direction, True))   # Second, with placing a bomb
        
        actions.append(((0, 0), True))
            
        # Filter the actions based on whether the move is valid or not
        actions = [(direction, place_bomb) for direction, place_bomb in actions if self.is_valid_move(wrld, state, direction)]
        print(actions)

        return actions



    def is_valid_move(self, wrld, state, action):
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        return 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx,ny) and not wrld.explosion_at(nx,ny) and not wrld.monsters_at(nx,ny)


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
                total_cost = g_score[current] + 1
                return total_cost
            
            for neighbor in self.get_neighbors(wrld, current):
                temp_g_score = g_score[current] + 1 #+ self.get_proximity_cost(neighbor)
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return 40 # No path found

    def calculate_danger_grid(self, wrld):
        """Calculates the proximity of monsters for all cells in the world."""
        self.monster_present = False
        self.monster_positions = []
        # Initialize the danger grid with a high value
        danger_grid = { (x, y): float('inf') for x in range(wrld.width()) for y in range(wrld.height()) }

        # Initialize BFS queue and set proximity for monster positions
        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.monsters_at(x, y):  # If there is a monster in the cell
                    danger_grid[(x, y)] = 0  # The monster cell itself has proximity 0
                    queue.append((x, y))  # Add to BFS queue
                    self.monster_positions.append((x,y))
                    self.monster_present = True
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # BFS to propagate danger levels
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and danger_grid[(nx, ny)] == float('inf'):
                    danger_grid[(nx, ny)] = danger_grid[(x, y)] + 1
                    queue.append((nx, ny))
                    if danger_grid[(nx, ny)] == 1:
                        self.monster_positions.append((x,y))
                    if danger_grid[(nx, ny)] == 4:  # Stop expanding beyond 4 cells
                        continue
        
        return danger_grid
    

    def calculate_explosion_grid(self, wrld):
        self.explosion_present = False
        explosion_grid = { (x, y): .25 for x in range(wrld.width()) for y in range(wrld.height()) }

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
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and explosion_grid[(nx, ny)] == .25:
                    explosion_grid[(nx, ny)] = .5
        
        return explosion_grid
    
    def calculate_bomb_grid(self, wrld):
        explosion_grid = { (x, y): 0 for x in range(wrld.width()) for y in range(wrld.height()) }

        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):  
                    explosion_grid[(x, y)] = 1 
                    queue.append((x, y)) 

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
        

    def get_explosion_positions(self, wrld):
        self.explosion_positions = []

        queue = deque()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):  
                    self.explosion_positions.append((x, y)) 
                    queue.append((x,y))
                if wrld.explosion_at(x,y):
                    self.explosion_positions.append((x, y)) 

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
                    if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                        self.explosion_positions.append((x, y)) 
        
        return
    
    def get_proximity_cost(self, wrld, state):
        """Returns the proximity cost for a given cell (x, y)."""
        if not self.is_valid_move(wrld, state, (0,0)):
            return .1
        dist = self.danger_grid[state]
        
        if dist == 0:  # Monster cell itself (wall)
            return 1
        elif dist == 1:  # Cells around the monster (immediate danger)
            return 2  # Wall-like behavior
        elif dist == 2:  # Two cells away from monster (heavy penalty)
            return 3  # Heavy penalty
        elif dist == 3:  # Three cells away from monster (light penalty)
            return 4  # Light penalty
        elif dist == 4:  # Four cells away from monster (slight penalty)
            return 5  # Light penalty
        return 6  # Default cost for free space
    
        
    def get_bomb_cost(self, wrld, state):
        bomb_grid  = self.calculate_bomb_grid(wrld)
        """Returns the proximity cost for a given cell (x, y)."""
        if not self.is_valid_move(wrld, state, (0,0)) or bomb_grid[state] == 1:
            return 1
        return 5  # Default cost for free space
    
    

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
        wall_score = 10
        edge_score = 10
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