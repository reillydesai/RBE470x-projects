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

class TestCharacter(CharacterEntity):

    turn_counter = 0
    bomb1_dropped = False
    bomb2_dropped = False
    chased = False
    bomb_location = None

    def do(self, wrld):

        self.turn_counter += 1
        me = wrld.me(self)  # Get current character state
        start = (me.x, me.y)  # Get character's starting position
        monster_distance = self.get_proximity_cost(wrld, me.x, me.y)
            
        if monster_distance > 300:  # Monster is dangerously close
            print("FUCKKK")
            dx, dy = self.find_best_retreat(wrld, start)
            print(dx, dy)
            self.move(dx, dy)
            self.chased = True
            return
            
        self.chased = False

        # Define hardcoded bomb and exit locations
        bomb1 = (wrld.width() - 4, wrld.height() - 13)  # Example bomb1 location
        bomb2 = (wrld.width() - 1, wrld.height() - 5)   # Example bomb2 location
        exit_point = (wrld.width() - 1, wrld.height() - 1)  # Exit point

        # Get the A* path and f(n) for each location
        if not self.bomb1_dropped: 
            path_to_bomb1, cost_bomb1 = self.a_star_with_cost(wrld, start, bomb1)
        else:
            cost_bomb1 = float('inf')
            path_to_bomb1 = None
        if not self.bomb2_dropped and self.turn_counter > 12: 
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


        if start == goal and goal == bomb1:
            self.bomb1_dropped = True
            self.place_bomb()
            self.turn_counter = 0
            self.bomb_location = bomb1
        if start == goal and goal == bomb2 and self.turn_counter > 15:
            self.bomb2_dropped = True
            self.place_bomb()
            self.bomb_location = bomb2 
        elif path and len(path) > 0:
                next_move = path[0]  # Get the next step in the path
                dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
                self.move(dx, dy)  # Move in the determined direction


        print(f"Goal: {goal}")


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
                if goal == (wrld.width() - 1, wrld.height() - 1):
                    total_cost -= 1
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
                temp_g_score = g_score[current] + 1 + self.get_proximity_cost(wrld, *neighbor) + self.get_bomb_cost(wrld, *neighbor)
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None # No path found

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
            return 1000  # Wall-like behavior
        elif dist == 2:  # Two cells away from monster (heavy penalty)
            return 500  # Heavy penalty
        elif dist == 3:  # Three cells away from monster (light penalty)
            return 300  # Light penalty
        elif dist == 4:  # Four cells away from monster (slight penalty)
            return 100  # Light penalty
        return 1  # Default cost for free space
    
    def get_bomb_cost(self, wrld, x, y):
        if self.turn_counter > 12 and self.turn_counter < 16:
            bx, by = self.bomb_location
            if bx == x or bx == y:
                if abs(bx - x) < 6 or abs(by - y) < 6:
                    return 800 
        return 0

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
        """Finds the best retreat direction away from the nearest monster."""
       
        px, py = start
        queue = deque([(px, py)])
        visited = set()
        closest_monster = None
        closest_distance = float('inf')

        # Pre-populate the queue with cells in a 3-cell radius in all directions from the player
        for dx in range(-3, 4):  # -3 to 3
            for dy in range(-3, 4):  # -3 to 3
                nx, ny = px + dx, py + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and (nx, ny) != (px, py):
                    queue.append((nx, ny))

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # If a monster is found, determine the retreat direction
            if wrld.monsters_at(x, y):
                distance = abs(px - x) + abs(py - y)

                # If this monster is closer than the previous closest one, update the closest monster
                if distance < closest_distance:
                    closest_distance = distance
                    closest_monster = (x, y)
            
        if closest_monster:    
            mx, my = closest_monster
            dx, dy = px - mx, py - my  # Reverse direction away from the monster

            # Normalize dx and dy to -1, 0, or 1
            norm_dx = 0 if dx == 0 else (1 if dx > 0 else -1)
            norm_dy = 0 if dy == 0 else (1 if dy > 0 else -1)

            # Ensure the retreat cell is walkable
            retreat_x, retreat_y = px + norm_dx, py + norm_dy
            print(f"Attempting to retreat in direction ({norm_dx}, {norm_dy})")

            if not (0 <= retreat_x < wrld.width() and 0 <= retreat_y < wrld.height()) or wrld.wall_at(retreat_x, retreat_y):
                # If blocked by a wall, attempt side-stepping or another direction
                print(f"Retreat blocked by a wall. Trying alternative directions...")

                best_direction = (0, 0)
                best_score = float('-inf')  # Start with an impossibly low score

                
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)] 

                # Try side-stepping: if blocked, attempt other directions
                for side_dx, side_dy in directions:
                    alt_retreat_x, alt_retreat_y = px + side_dx, py + side_dy
                    if 0 <= alt_retreat_x < wrld.width() and 0 <= alt_retreat_y < wrld.height():
                        if not wrld.wall_at(alt_retreat_x, alt_retreat_y):

                            score = 0

                            # Calculate distance to the monster (lower score for being closer)
                            dist_to_monster = abs(alt_retreat_x - mx) + abs(alt_retreat_y - my)
                            score += dist_to_monster  # The closer to the monster, the worse
                            
                            # Check if it's near a wall (avoid getting trapped)
                            score += self.wall_proximity(wrld, alt_retreat_x, alt_retreat_y)
                

                            # Update best direction if this direction has a better score
                            if score > best_score:
                                best_score = score
                                best_direction = (side_dx, side_dy)


                norm_dx, norm_dy = best_direction
                print(f"Side-step successful. Retreating to ({norm_dx}, {norm_dy})")
                            

        

            return norm_dx, norm_dy 

        return 0, 0    
    
    def wall_proximity(self, wrld, x, y):
        wall_score = 1
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