# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import heapq
import math
from collections import deque

class TestCharacter(CharacterEntity):

    state = 0
    turn_counter = 0
    bomb1_dropped = False
    bomb2_dropped = False

    def do(self, wrld):

        me = wrld.me(self)  # Get current character state
        start = (me.x, me.y)  # Get character's starting position
        monster_distance = self.get_proximity_cost(wrld, me.x, me.y)
            
        if monster_distance > 10:  # Monster is dangerously close
            goal = self.find_best_retreat(wrld, start)
            path = self.a_star(wrld, start, goal)  # Run away to top-left
            if path and len(path) > 0:
                    next_move = path[0]  # Get the next step in the path
                    dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
                    self.move(dx, dy)  # Move in the determined direction
                    return
            
        # Define hardcoded bomb and exit locations
        bomb1 = (wrld.width() - 4, wrld.height() - 13)  # Example bomb1 location
        bomb2 = (wrld.width() - 2, wrld.height() - 5)   # Example bomb2 location
        exit_point = (wrld.width() - 1, wrld.height() - 1)  # Exit point

        # Get the A* path and f(n) for each location
        if not self.bomb1_dropped: 
            path_to_bomb1, cost_bomb1 = self.a_star_with_cost(wrld, start, bomb1)
        else:
            cost_bomb1 = float('inf')
            path_to_bomb1 = None
        if not self.bomb2_dropped: 
            path_to_bomb2, cost_bomb2 = self.a_star_with_cost(wrld, start, bomb2)
        else:
            cost_bomb2 = float('inf')
            path_to_bomb2 = None
        path_to_exit, cost_exit = self.a_star_with_cost(wrld, start, exit_point)

        # Find the location with the cheapest path (minimum cost)
        min_cost = min(cost_bomb1, cost_bomb2, cost_exit)

        if min_cost == cost_bomb1:
            path = path_to_bomb1
            goal = bomb1
        elif min_cost == cost_bomb2:
            path = path_to_bomb2
            goal = bomb2
        else:
            path = path_to_exit
            goal = exit_point


        if self.state == 0:
            if start == goal and goal == bomb1:
                self.bomb1_dropped = True
                self.place_bomb()
                #self.state = 1
            if start == goal and goal == bomb2:
                self.bomb2_dropped = True
                self.place_bomb()
                #self.state = 1   
            elif path and len(path) > 0:
                    next_move = path[0]  # Get the next step in the path
                    dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
                    self.move(dx, dy)  # Move in the determined direction
        elif self.state == 1:
            goal = self.find_best_retreat(wrld, start)
            path = self.a_star(wrld, start, goal) 
            if path and len(path) > 0:
                    next_move = path[0]  # Get the next step in the path
                    dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
                    self.move(dx, dy)  # Move in the determined direction


            self.turn_counter += 1
            if self.turn_counter >= 1: #15
                self.state = 0
                self.turn_counter = 0


        print(f"State: {self.state}, Goal: {goal}")

            # self.escape_mode = False
            # if (self.state == 0):
            #     goal = (wrld.width() - 4, wrld.height() - 13)  # Exit is always at the bottom-right corner
            
            #     path = self.a_star(wrld, start, goal)  # Compute A* path to the goal
            
            #     if start == goal:
            #         self.place_bomb()
            #         self.state = 1

            #     elif path and len(path) > 0:
            #         next_move = path[0]  # Get the next step in the path
            #         dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            #         self.move(dx, dy)  # Move in the determined direction
            # if (self.state == 1):

            #     goal = self.find_best_retreat(wrld, start)
            #     path = self.a_star(wrld, start, goal)  # Run away to top-left
            #     if path and len(path) > 0:
            #             next_move = path[0]  # Get the next step in the path
            #             dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            #             self.move(dx, dy)  # Move in the determined direction


            #     self.turn_counter += 1
            #     if self.turn_counter >= 15:
            #         self.state = 2
            # if (self.state == 2):
            
            #     goal = (wrld.width() - 2, wrld.height() - 5)  # Exit is always at the bottom-right corner
            
            #     path = self.a_star(wrld, start, goal)  # Compute A* path to the goal

            #     if start == goal:
            #         self.place_bomb()
            #         self.state = 3

            #     if path and len(path) > 0:
            #         next_move = path[0]  # Get the next step in the path
            #         dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            #         self.move(dx, dy)  # Move in the determined direction
            # if (self.state == 3):
                    
            #     #goal = (wrld.width() - 3, wrld.height() - 15)  # Exit is always at the bottom-right corner
            #     # goal = (0,0)

            #     # path = self.a_star(wrld, start, goal)  # Compute A* path to the goal

            #     # if path and len(path) > 0:
            #     #     next_move = path[0]  # Get the next step in the path
            #     #     dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            #     #     self.move(dx, dy)  # Move in the determined direction


            #     goal = self.find_best_retreat(wrld, start)
            #     path = self.a_star(wrld, start, goal)  # Run away to top-left
            #     if path and len(path) > 0:
            #             next_move = path[0]  # Get the next step in the path
            #             dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            #             self.move(dx, dy)  # Move in the determined direction





            #     self.turn_counter += 1
            #     if self.turn_counter >= 15:
            #         self.state = 4
            # if (self.state == 4):
            
            #     goal = (wrld.width() - 1, wrld.height() - 1)  # Exit is always at the bottom-right corner
            
            #     path = self.a_star(wrld, start, goal)  # Compute A* path to the goal

            #     if path and len(path) > 0:
            #         next_move = path[0]  # Get the next step in the path
            #         dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            #         self.move(dx, dy)  # Move in the determined direction


        # """Determines the next move for the character using A* pathfinding."""
        # me = wrld.me(self)  # Get current character state
        # start = (me.x, me.y)  # Get character's starting position
        # goal = (wrld.width() - 1, wrld.height() - 1)  # Exit is always at the bottom-right corner
        
        # path = self.a_star(wrld, start, goal)  # Compute A* path to the goal
        
        # if path and len(path) > 0:
        #     next_move = path[0]  # Get the next step in the path
        #     dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
        #     self.move(dx, dy)  # Move in the determined direction


    def a_star_with_cost(self, wrld, start, goal):
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
                return self.reconstruct_path(wrld, came_from, current), total_cost 
            
            for neighbor in self.get_neighbors(wrld, current):
                # g(n) = Manhattan cost (1) + proximity to monster (max 3 cells)
                temp_g_score = g_score[current] + 1 + self.get_proximity_cost(wrld, *neighbor)
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, float('inf') # No path found

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
                return self.reconstruct_path(wrld, came_from, current)
            
            for neighbor in self.get_neighbors(wrld, current):
                # g(n) = Manhattan cost (1) + proximity to monster (max 3 cells)
                temp_g_score = g_score[current] + 1 + self.get_proximity_cost(wrld, *neighbor)
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None# No path found

    def calculate_monster_proximity(self, wrld):
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
        
        # Directions for BFS: up, down, left, right
        #directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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

    def get_proximity_cost(self, wrld, x, y):
        """Returns the proximity cost for a given cell (x, y)."""
        danger_grid = self.calculate_monster_proximity(wrld)
        dist = danger_grid[(x, y)]
        
        if dist == 0:  # Monster cell itself (wall)
            return float('inf')
        elif dist == 1:  # Cells around the monster (immediate danger)
            return 100  # Wall-like behavior
        elif dist == 2:  # Two cells away from monster (heavy penalty)
            return 50  # Heavy penalty
        elif dist == 3:  # Three cells away from monster (light penalty)
            return 30  # Light penalty
        elif dist == 4:  # Four cells away from monster (slight penalty)
            return 10  # Light penalty
        return 1  # Default cost for free space

    def heuristic(self, pos, goal):
        """Euclidean distance heuristic."""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def get_neighbors(self, wrld, pos):
        """Returns valid neighboring positions (up, down, left, right) ignoring walls."""
        x, y = pos
        neighbors = []
        #directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        danger_grid = self.calculate_monster_proximity(wrld)  # Get monster proximity map
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny):
                neighbors.append((nx, ny))
        
        # If in escape mode, prioritize the SAFEST move, even if it's away from the goal
        # if self.escape_mode:
        #     neighbors.sort(key=lambda cell: danger_grid[cell], reverse=True)  # Move to safest cell


        return neighbors

    def reconstruct_path(self, wrld, came_from, current):
        """Reconstructs the path from goal to start."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        self.visualize_path(wrld, path)
        
        return path
    
    def visualize_path(self, wrld, path):
        """Resets all cells to default color."""
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                self.set_cell_color(x, y, Fore.WHITE + Back.BLACK)  # Default color (white text on black background)

        """Visualizes the path by marking each cell with a color."""
        for x, y in path:
            # Mark the cell with a color (red text on green background)
            self.set_cell_color(x, y, Fore.RED + Back.GREEN)

    def find_best_retreat(self, wrld, start):
        # """Finds the safest and most reachable retreat location by selecting the neighbor
        # that puts the character farthest away from the monster's neighbors."""
        # # Get the 8 neighbors of the current position
        # my_neighbors = self.get_neighbors(wrld, start)
        
        # # Get all the monster neighbors
        # monster_neighbors = []
        # for x in range(wrld.width()):
        #     for y in range(wrld.height()):
        #         if wrld.monsters_at(x, y):  # If there's a monster at this position
        #             monster_neighbors.extend(self.get_neighbors(wrld, (x, y)))  # Add the monster's neighbors

        # # Calculate the best retreat by choosing the neighbor that maximizes distance from monster neighbors
        # best_neighbor = None
        # max_distance = -1
        
        # for neighbor in my_neighbors:
        #     # Find the minimum distance from this neighbor to any of the monster's neighbors
        #     min_distance_to_monster = min([self.heuristic(neighbor, monster_neighbor) for monster_neighbor in monster_neighbors])
            
        #     # We want the neighbor that maximizes this minimum distance
        #     if min_distance_to_monster > max_distance:
        #         max_distance = min_distance_to_monster
        #         best_neighbor = neighbor
        
        # # Return the best neighbor, or the start if no valid retreat is found
        # return best_neighbor if best_neighbor else start
        print("FUCKKKKK")

        px, py = start
        danger_grid = self.calculate_monster_proximity(wrld)

        # BFS to find the nearest safe cell
        queue = deque([(px, py)])
        visited = set()

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if danger_grid.get((x, y), 0) == float('inf'):
                # Move in the opposite direction of the safe cell
                dx, dy = px - x, py - y
                return px + (dx and (1 if dx < 0 else -1)), py + (dy and (1 if dy < 0 else -1))

            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    queue.append((nx, ny))

        return px, py

            