import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Node class representing a state in the space
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

# RRT* algorithm
class RRTStar:
    def __init__(self, start, goal, obstacles, step_size=0.2, max_iter=800):
        random.seed(80)  # For reproducibility
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles

        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]

        self.goal_radius = 0.3
        self.robot_radius = 0.4
        
        self.search_radius = 3.0
        self.path = None
        self.goal_reached = False
        

        # Visualization setup
        self.xlim = (-3, 3)
        self.ylim = (-3, 3)
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.setup_visualization()

        self.gamma = self.calculate_gamma()
        self.dim = 2

        self.best_goal_node = None
        self.best_cost = float('inf') 

    def setup_visualization(self):
        #self.ax.grid(True,color='black', linestyle='--', linewidth=0.5)
        self.ax.set_xlim(self.xlim[0]-0.5, self.xlim[1]+0.5)
        self.ax.set_ylim(self.ylim[0]-0.5, self.ylim[1]+0.5)        
        self.draw_obstacles()
        
        
    def calculate_gamma(self):
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]
        area_total = width * height

        d = 2

        zeta_d = math.pi
        
        term1 = 2 * (1 + 1/d)**(1/d) 
        term2 = (area_total / zeta_d)**(1/d) 
        
        gamma = 2 * term1 * term2
        
        return gamma
    
    def draw_obstacles(self):
        for i, (ox, oy, size, rsafe) in enumerate(self.obstacles):
            rsafe = rsafe
            if i == 0:
                label1 = 'Robot augmented radius'
                label2 = 'Static obstacle safe radius'
                label3 = 'Static obstacle'
            else:
                label1 = None
                label2 = None
                label3 = None
            circle_safe_robot = plt.Circle((ox, oy), size + rsafe + self.robot_radius, ec='b', linestyle='dotted',alpha=0.9,label=label1,fill=False)
            circle_safe = plt.Circle((ox, oy), size + rsafe, alpha=0.1,fc='m', ec='m', linestyle='--',label=label2)
            circle = plt.Circle((ox, oy), size, fc='gray',ec='m',label=label3)
            
            self.ax.add_artist(circle)
            self.ax.add_artist(circle_safe)
            self.ax.add_artist(circle_safe_robot)

    # ---------------------- Sampling ---------------------- 
    def get_random_node(self):    
        if random.random() > 0.2: # 80% chance to sample random point
            rand_node = Node(random.uniform(self.xlim[0], self.xlim[1]), 
                             random.uniform(self.ylim[0], self.ylim[1]))  
        else:
            rand_node = Node(self.goal.x, self.goal.y)
        return rand_node
    
    # ---------------------- Nearest node ---------------------- 
    def nearest(self, rand_node):
        distances = [(node.x - rand_node.x)**2 + (node.y - rand_node.y)**2 for node in self.node_list]
        return self.node_list[np.argmin(distances)]
    
    # ---------------------- Steer ---------------------- 
    def steer(self, xnear, xrand):
        theta = math.atan2(xrand.y - xnear.y, xrand.x - xnear.x)
        new = Node(
            xnear.x + self.step_size * math.cos(theta),
            xnear.y + self.step_size * math.sin(theta))
        new.parent = xnear
        new.cost = xnear.cost + self.step_size
        return new

    # ---------------------- Collision checking point ---------------------- 
    def collision_free_point(self, n):
        for (ox, oy, size, rsafe) in self.obstacles:
            r = size + rsafe + self.robot_radius
            if (n.x - ox)**2 + (n.y - oy)**2 <= r*r:
                return False
        return True

    # ---------------------- Find neighbors ---------------------- 
    def near(self, xnew):
        n = len(self.node_list)
        n_eff = max(2, n)
        r = min(self.gamma * (math.log(n_eff) / n_eff) ** (1.0 / self.dim), self.search_radius)
        neighbors = []
        for node in self.node_list:
            if (node.x - xnew.x)**2 + (node.y - xnew.y)**2 <= r*r:
                neighbors.append(node)
        return neighbors

    # ---------------------- In configuration space ----------------------
    def in_conf_space(self, node):
        return (self.xlim[0] < node.x < self.xlim[1] and 
                self.ylim[0] < node.y < self.ylim[1])
   
    # ---------------------- Collision checking edge ---------------------- 
    def collision_free_edge(self, n1, n2):
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        L2 = dx*dx + dy*dy
        if L2 < 1e-12:
            return self.collision_free_point(n1)
        for (ox, oy, size, rsafe) in self.obstacles:
            r = size + rsafe + self.robot_radius
            r2 = r*r

            fx = ox - n1.x
            fy = oy - n1.y

            t = max(0, min(1, (fx*dx + fy*dy) / L2))
            cx = n1.x + t*dx
            cy = n1.y + t*dy

            if (cx - ox)**2 + (cy - oy)**2 <= r2:
                return False
        return True

    # ---------------------- Assign cost and parent ----------------------
    def choose_parent(self, neighbors, xnew):
        best_parent = None
        best_cost = float('inf')

        for n in neighbors:
            d = math.hypot(n.x - xnew.x, n.y - xnew.y)
            new_cost = n.cost + d
            if new_cost < best_cost and self.collision_free_edge(n, xnew):
                best_cost = new_cost
                best_parent = n

        if best_parent is None:
            return None  

        xnew.cost = best_cost
        xnew.parent = best_parent
        return xnew

    # ---------------------- Rewiring ----------------------
    def rewire(self, xnew, neighbors):
        for n in neighbors:
            if n is xnew.parent:
                continue
            d = math.hypot(n.x - xnew.x, n.y - xnew.y)
            new_cost = xnew.cost + d
            if new_cost < n.cost and self.collision_free_edge(xnew, n):
                n.parent = xnew
                n.cost = new_cost

    # ---------------------- Goal check ----------------------
    def reached_goal(self, node):
        return math.hypot(node.x - self.goal.x, node.y - self.goal.y) < self.goal_radius
        

    # ---------------------- Generate final path ----------------------
    def extract_path(self, node):
        path = []
        while node:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]

    # ---------------------- Draw final path ----------------------
    def draw_path(self):
        if self.path:
            full_path = self.path + [[self.goal.x, self.goal.y]]
            self.ax.plot([x[0] for x in full_path], [x[1] for x in full_path], '-r', linewidth=2, label='Final Path')
    
    # ---------------------- Main planning function ----------------------
    def plan(self):
            for _ in range(self.max_iter):
                xrand = self.get_random_node()
                xnear = self.nearest(xrand)
                xnew = self.steer(xnear, xrand)

                if not self.collision_free_point(xnew) or not self.in_conf_space(xnew):
                    continue

                neighbors = self.near(xnew)

                xnew = self.choose_parent(neighbors,xnew)
                if xnew is None:
                    continue

                self.node_list.append(xnew)

                self.rewire(xnew, neighbors)

                if self.reached_goal(xnew):
                    if xnew.cost < self.best_cost:
                        self.best_cost = xnew.cost
                        self.best_goal_node = xnew
                        print(f"New best cost at iteration {_}: {self.best_cost}")

            if self.best_goal_node:
                return self.extract_path(self.best_goal_node)
            return None



# Main execution
def planner(start, goal, obstacles):
    rrt_star = RRTStar(start, goal, obstacles)
    to= time.time()
    path = rrt_star.plan()
    tf= time.time()
    print("Planning time:", tf - to)
    if path is not None:
        rrt_star.path = path
        waypoints = path + [goal]

        print("Final Cost:", rrt_star.best_cost)

        for j,node in enumerate(rrt_star.node_list):
            if node.parent:
                if j == 1:
                    labelr = 'RRT* Tree'
                else: 
                    labelr = None
                rrt_star.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], color="y",alpha=0.5, label=labelr)
        rrt_star.draw_path()
        plt.scatter(*zip(*waypoints), c='r', marker='o', label='Waypoints')
        rrt_star.ax.plot(rrt_star.start.x, rrt_star.start.y, 'bo', label='Start', markersize=8)
        rrt_star.ax.plot(rrt_star.goal.x, rrt_star.goal.y, "m*", label='Goal', markersize=15)
        rrt_star.ax.set_ylabel("y (m)")
        rrt_star.ax.set_xlabel("x (m)")
        rrt_star.fig.legend(loc='outside upper center', ncol=4,borderaxespad=3)
        plt.show()
        return waypoints

    else:
        print("No path found")
        plt.show()
        return []

# Example usage:
# r = 0.4
# rsafe = 0.20
# S_obstacle = [(-1.5, 1.0, r, rsafe), (-0.5, -1.0, r, rsafe),(1.0, 1.0, r, rsafe),(1.8, -1.5, r, rsafe)] 
# start = [-2, -1]
# goal = [2.5,1]
# waypoints = planner(start, goal, S_obstacle)