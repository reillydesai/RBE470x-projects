# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

import heapq
import math
from collections import deque

class InteractiveCharacter(CharacterEntity):

    def do(self, wrld):
        # Commands
        # dx, dy = 0, 0
        # bomb = False
        # # Handle input
        # for c in input("How would you like to move (w=up,a=left,s=down,d=right,b=bomb)? "):
        #     if 'w' == c:
        #         dy -= 1
        #     if 'a' == c:
        #         dx -= 1
        #     if 's' == c:
        #         dy += 1
        #     if 'd' == c:
        #         dx += 1
        #     if 'b' == c:
        #         bomb = True
        # Execute commands
        # self.move(dx, dy)
        # if bomb:
        #     self.place_bomb()
        #for c in input("Advance? (y)"):

        ## Grab map

        ## Calculate A*
        # g(n) = manhattan distance + proximity to monster (max check: 3 cells)
        # h(n) = euclidean distance

        ## Which direction is the next move in the path? 

        ## Move that way 
        # self.move()

        """Determines the next move for the character using A* pathfinding."""
        me = wrld.me(self)  # Get current character state
        start = (me.x, me.y)  # Get character's starting position
        goal = (wrld.width() - 1, wrld.height() - 1)  # Exit is always at the bottom-right corner
        
        path = self.a_star(wrld, start, goal)  # Compute A* path to the goal
        
        if path and len(path) > 0:
            next_move = path[0]  # Get the next step in the path
            dx, dy = next_move[0] - start[0], next_move[1] - start[1]  # Calculate movement vector
            self.move(dx, dy)  # Move in the determined direction


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
        
        return None  # No path found

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
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # BFS to propagate danger levels
        while queue:
            x, y = queue.popleft()
            
            # If we're already at distance 3, stop expanding further
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and danger_grid[(nx, ny)] == float('inf'):
                    danger_grid[(nx, ny)] = danger_grid[(x, y)] + 1
                    queue.append((nx, ny))
        
        return danger_grid

    def get_proximity_cost(self, wrld, x, y):
        """Returns the proximity cost for a given cell (x, y)."""
        danger_grid = self.calculate_monster_proximity(wrld)
        dist = danger_grid[(x, y)]
        
        if dist == 0:  # Monster cell itself (wall)
            return float('inf')
        elif dist == 1:  # Cells around the monster (immediate danger)
            return 10  # Wall-like behavior
        elif dist == 2:  # Two cells away from monster (heavy penalty)
            return 5  # Heavy penalty
        elif dist == 3:  # Three cells away from monster (light penalty)
            return 2  # Light penalty
        return 1  # Default cost for free space

    def heuristic(self, pos, goal):
        """Euclidean distance heuristic."""
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def get_neighbors(self, wrld, pos):
        """Returns valid neighboring positions (up, down, left, right) ignoring walls."""
        x, y = pos
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny):
                neighbors.append((nx, ny))
        
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
        