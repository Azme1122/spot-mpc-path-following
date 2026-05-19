import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from matplotlib.transforms import Affine2D

def animate_spot(robot, start, goal, path_robot, trajectory_robot, S_obstacle, D_obstacle, path_o, ppath_o, dt,name):

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15,metadata=dict(artist='Me'),bitrate=1800)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(path_robot[0], path_robot[1], 'k--', label='Reference Path')

    #-------------- Spot dimensions --------------
    width, height = 1.1, 0.5
    radius = robot[0] 
    head_radius = 0.05
    circle_offset = robot[1] # offset from center of body to each circle's center

    #-------------- Draw Spot centered at (0,0) --------------
    spot_body = patches.Rectangle((-width/2, -height/2), width, height, fc='blue', ec='black') 
    spot_head = patches.Circle((0, 0), radius=head_radius, fc='red', ec='none')

    circle_1 = patches.Circle((0, 0), radius, alpha=0.3, fc='blue')
    circle_2 = patches.Circle((0, 0), radius, alpha=0.3, fc='blue')
    line, = ax.plot([], [], 'r-', linewidth=3)

    # -------------- Dynamic obstacle --------------
    obs1 = patches.Circle((D_obstacle[0], D_obstacle[1]), D_obstacle[2], alpha=0.8, fc='m')
    safe_obs1 = patches.Circle((D_obstacle[0], D_obstacle[1]),D_obstacle[2] + D_obstacle[3], alpha=0.1, fc='m',linestyle='--')

    # -------------- Static obstacles --------------
    static_obs = []
    safe_obs = []
    for obs in S_obstacle:
        static_obs.append(patches.Circle((obs[0], obs[1]), obs[2], fc='gray',ec='m'))
        ax.add_patch(static_obs[-1])
        safe_obs.append(patches.Circle((obs[0], obs[1]), obs[2]+obs[3], alpha=0.1,fc='m', ec='m', linestyle='--'))
        ax.add_patch(safe_obs[-1])

    # -------------- Add patches --------------
    ax.add_patch(spot_body)
    ax.add_patch(spot_head)
    ax.add_patch(circle_1)
    ax.add_patch(circle_2)
    ax.add_patch(obs1)
    ax.add_patch(safe_obs1)

    # -------------- Trajectory lines and text --------------
    traj_line, = ax.plot([], [], 'b-', label='Spot Trajectory')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    obs_line, = ax.plot([], [], 'm--', label='Dynamic Obstacle Trajectory')
    ppath_line, = ax.plot([], [], 'x', label='Predicted Obstacle Trajectory')

    # -------------- Axes setup --------------
    ax.plot(start[0], start[1], 'bo', label='Start',markersize=10)
    ax.plot(goal[0], goal[1], "m*", label='Goal',markersize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('Spot MPC Path Following')

    def init():
        traj_line.set_data([], [])
        obs_line.set_data([], [])
        line.set_data([], [])
        ppath_line.set_data([], [])

        spot_body.set_transform(Affine2D().translate(-1e3, -1e3) + ax.transData)
        spot_head.center = (-1e3, -1e3)

        circle_1.set_center((0, 0))
        circle_2.set_center((0, 0))

        obs1.set_center((0, 0))
        safe_obs1.set_center((0, 0))

        time_text.set_text('')
        return traj_line, spot_body, spot_head, circle_1, circle_2, time_text, line, obs_line, obs1, safe_obs1

    def animate(i):
        if i >= len(trajectory_robot[0]):
            return traj_line, spot_body, spot_head, circle_1, circle_2, time_text, line, obs_line, obs1, safe_obs1

        x = trajectory_robot[0][i]
        y = trajectory_robot[1][i]
        theta = trajectory_robot[2][i]

        #-------------- Updating position of Dynamic Obstacle --------------
        patho_x = path_o[0]
        patho_y = path_o[1]
        ox = patho_x[i]
        oy = patho_y[i]

        # -------------- Spot and Dynamic Obstacle Trajectories --------------
        traj_line.set_data(trajectory_robot[0][:i + 1], trajectory_robot[1][:i + 1])
        obs_line.set_data(patho_x[:i + 1], patho_y[:i + 1])
        ppath_line.set_data(ppath_o[0][:i + 1], ppath_o[1][:i + 1])

        # -------------- Dynamic Obstacle position --------------
        obs1.set_center((ox, oy))
        safe_obs1.set_center((ox, oy))

        # -------------- Move robot: rotate then translate --------------
        # rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # rotate about center, then translate to (cx,cy)
        T = Affine2D().rotate(theta).translate(x, y)
        spot_body.set_transform(T + ax.transData)

        # head at the FRONT EDGE of the body
        hx = x + (width/2) * cos_theta
        hy = y + (width/2) * sin_theta
        spot_head.center = (hx, hy)
        
        # -------------- Robot circles position --------------
        circle1_pos = np.array([x - circle_offset*np.cos(theta), y - circle_offset*np.sin(theta)])
        circle2_pos = np.array([x + circle_offset*np.cos(theta), y + circle_offset*np.sin(theta)])

        circle_1.set_center(circle1_pos)
        circle_2.set_center(circle2_pos)
        line.set_data([circle1_pos[0], circle2_pos[0]], [circle1_pos[1], circle2_pos[1]])

        # -------------- Update time text --------------
        time_text.set_text(f"t = {i * dt:.2f} s")

        return traj_line, spot_body, spot_head, circle_1, circle_2, time_text, line, obs_line, obs1, safe_obs1

    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory_robot[0]), init_func=init,
        interval=63, blit=True, repeat=True)
    ani.save(name+".mp4",writer= writer)

    plt.show()
    return ani
