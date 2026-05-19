import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

def plot_robot_circles(robot_x, robot_y, rect_angle,circle1_center, circle1_radius, circle2_center, circle2_radius,
                                    obstacles, start_point, goal_point,figsize=(10, 10)):
    
    rect_width = 0.5 
    rect_height = 1.1
    fig, ax = plt.subplots(figsize=figsize)
    


    rect_x = robot_x - rect_width/2
    rect_y = robot_y - rect_height/2
    
    # Create and plot robot rectangle (body)
    robot_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                          angle=rect_angle, 
                          fill=True, color='lightblue', alpha=0.7, 
                          linewidth=2,
                          label='Robot Body')
    t = transforms.Affine2D().rotate_deg(rect_angle).translate(robot_x, robot_y) + ax.transData
    robot_rect.set_transform(t)
    ax.add_patch(robot_rect)

    # Plot robot position as red X
    ax.plot(robot_x, robot_y, 'rx', markersize=15, markeredgewidth=4, label='Robot Position')
    
    # Create and plot robot circles
    circle1 = plt.Circle(circle1_center, circle1_radius, 
                        fill=False, color='blue', linewidth=3, label='Robot Circle 1')
    ax.add_patch(circle1)
    
    circle2 = plt.Circle(circle2_center, circle2_radius,
                        fill=False, color='green', linewidth=3, label='Robot Circle 2')
    ax.add_patch(circle2)
    
    # Plot circle centers as dots
    ax.plot(circle1_center[0], circle1_center[1], 'bo', markersize=8)
    ax.plot(circle2_center[0], circle2_center[1], 'go', markersize=8)
    
    # Plot obstacles
    if obstacles is not None:
        for i, obs in enumerate(obstacles):
            ox, oy, orad, safety_margin = obs[0], obs[1], obs[2], obs[3] if len(obs) > 3 else 0.0
            # Plot obstacle core
            obs_circle = plt.Circle((ox, oy), orad, 
                                  fill=True, color='red', alpha=0.7, 
                                  label='Obstacle' if i == 0 else "")
            ax.add_patch(obs_circle)
            
            # Plot safety margin if exists
            if safety_margin > 0:
                safety_circle = plt.Circle((ox, oy), orad + safety_margin, 
                                         fill=False, color='orange', linewidth=2, linestyle='--',
                                         label='Safety Margin' if i == 0 else "")
                ax.add_patch(safety_circle)
            
            # Label obstacle center
            ax.plot(ox, oy, 'ko', markersize=6)
            ax.text(ox + 0.1, oy + 0.1, f'Obs {i+1}', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Plot start point
    if start_point is not None:
        ax.plot(start_point[0], start_point[1], 'gs', markersize=12, 
               markeredgewidth=2, markeredgecolor='darkgreen', label='Start')
        ax.text(start_point[0] + 0.1, start_point[1] + 0.1, 'START', fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Plot goal point
    if goal_point is not None:
        ax.plot(goal_point[0], goal_point[1], 'r^', markersize=12, 
               markeredgewidth=2, markeredgecolor='darkred', label='Goal')
        ax.text(goal_point[0] + 0.1, goal_point[1] + 0.1, 'GOAL', fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    arrow_length = rect_width * 0.4
    arrow_x = robot_x + arrow_length * np.cos(np.radians(rect_angle))
    arrow_y = robot_y + arrow_length * np.sin(np.radians(rect_angle))
    ax.annotate('', xy=(arrow_x, arrow_y), xytext=(robot_x, robot_y),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=3))

    # Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Calculate plot limits with padding
    all_x = [robot_x, circle1_center[0], circle2_center[0]]
    all_y = [robot_y, circle1_center[1], circle2_center[1]]
    
    # Add obstacle positions to bounds calculation
    if obstacles is not None:
        for obs in obstacles:
            all_x.extend([obs[0]])
            all_y.extend([obs[1]])
    
    # Add start/goal to bounds
    if start_point is not None:
        all_x.append(start_point[0])
        all_y.append(start_point[1])
    if goal_point is not None:
        all_x.append(goal_point[0])
        all_y.append(goal_point[1])
    
    # Calculate bounds with padding
    max_radius = max(circle1_radius, circle2_radius)
    if obstacles is not None:
        max_radius = max(max_radius, max([obs[2] + (obs[3] if len(obs) > 3 else 0) for obs in obstacles]))
    
    padding = max_radius + 0.5
    x_min, x_max = min(all_x) - padding, max(all_x) + padding
    y_min, y_max = min(all_y) - padding, max(all_y) + padding
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Labels and legend
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def postest(rx,ry,th):
    lcos = 0.4 * np.cos(th)
    lsin = 0.4 * np.sin(th)
    c1_pos = np.array([rx - lcos, ry - lsin])  # front circle
    c2_pos = np.array([rx + lcos, ry + lsin])  # rear circle

    print("Circle 1 position:", c1_pos)
    print("Circle 2 position:", c2_pos)
    return c1_pos, c2_pos