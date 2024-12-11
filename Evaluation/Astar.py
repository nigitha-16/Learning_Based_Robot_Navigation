import heapq
import numpy as np

class AStarPlanner:
    #  for movement (8 directions: up, down, left, right, and diagonals)
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals: top-left, top-right, bottom-left, bottom-right
    
    def __init__(self, grid):
        """Initialize the A* pathfinder with an occupancy grid."""
        self.grid = grid
        
    def is_valid_move(self, x, y):
        """Check if the move is within bounds and not an obstacle."""
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] == 0

    def heuristic(self, a, b):
        """Heuristic function for A* (Manhattan distance)."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1]) **2) # Chebyshev distance for diagonal movement

    def a_star_search(self, start, goal):
        """A* algorithm to find the shortest path from start to goal."""
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, start))  # (f, g, (x, y))
        came_from = {}
        g_costs = {start: 0}
        
        while open_list:
            _, g, current = heapq.heappop(open_list)
            if current == goal:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return reversed path
            
            for dx, dy in self.DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.is_valid_move(*neighbor):
                    if dx == dy:
                        tentative_g = g + 1.5  # Assume cost between nodes is 1
                    else:
                        tentative_g = g + 1 
                    if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g
                        f_cost = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_cost, tentative_g, neighbor))
                        came_from[neighbor] = current
        
        return []  # No path found
