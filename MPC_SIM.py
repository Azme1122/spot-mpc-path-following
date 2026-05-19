import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
from pygrampc import Grampc, GrampcResults
from Animation_obs import animate_spot
from path_planning import planner
from pos_predictor import K_Spot, K_Obs
from path_param import ParameterizedPath
from mpc_problem import Spot 
from utils import angle_wrap, xdot2vspot, CVM, vel_clip, min_circ_dist
import pandas as pd
import os
import time
import csv

#Import data from tsv files using pandas
data_name = 'CASE3.1'
data_dir = f'Comp/{data_name}'
spot_x = pd.read_csv(os.path.join(data_dir, f'{data_name}_x_s_m.tsv'), sep='\t', header=None)
spot_y = pd.read_csv(os.path.join(data_dir, f'{data_name}_y_s_m.tsv'), sep='\t', header=None)
spot_theta = pd.read_csv(os.path.join(data_dir, f'{data_name}_theta_s_m.tsv'), sep='\t', header=None)
obj_x = pd.read_csv(os.path.join(data_dir, f'{data_name}_x_o_m.tsv'), sep='\t', header=None)
obj_y = pd.read_csv(os.path.join(data_dir, f'{data_name}_y_o_m.tsv'), sep='\t', header=None)
# Convert to numpy arrays
spot_x = spot_x.to_numpy().flatten()
spot_y = spot_y.to_numpy().flatten()
spot_theta = spot_theta.to_numpy().flatten()
obj_x = obj_x.to_numpy().flatten()
obj_y = obj_y.to_numpy().flatten()

if __name__ == "__main__":
    name=f'{data_name}_sim'

    x0_m = np.array([spot_x[0], spot_y[0], spot_theta[0]])  # Initial measured position from OptiTrack

    # ---------------- Static obstacles ----------------
    r = 0.30
    rsafe = 0.20
    #S_obstacle = [(-1.19384074, -1.37350976, r, rsafe),(-1.19384074, -1.37350976, r, rsafe),(-1.19384074, -1.37350976, r, rsafe)] 
    #S_obstacle = [(-0.75, 0.13, r, rsafe),(1.15, -1.78, r, rsafe),(1.19384074, 1.37350976, r, rsafe), (-1.16, -1.26, r, rsafe),(-1.8, 0.22, r, rsafe)] #added 12.11
    #S_obstacle = [(-0.75, 0.13, r, rsafe), (-1.16, -1.26, r, rsafe),(-1.8, 0.22, r, rsafe)] #added 12.11
    S_obstacle = [(-1.0, 2.5, r, rsafe), (-0.0, 2.5, r, rsafe),(0.5,1.5, r, rsafe),(0.5,0.5, r, rsafe),(-0.0, 0.0, r, rsafe),(-1.0,0.0, r, rsafe)] #CASE 2 & 3
    # S_obstacle = []  #CASE 1

    # --------------- Dynamic obstacle ------------------
    # Initial state estimate for Kalman filter
    zobs_0 = np.array([obj_x[0], obj_y[0]])
    K_Obs.x = np.array([zobs_0[0], 0.0, zobs_0[1], 0.0])
    rdyn = 0.20
    D_obstacle = [zobs_0[0], zobs_0[1], rdyn, rsafe*1.4]

    # --------------- Waypoints & path ---------------
    # waypoints = [(x0_m[0],x0_m[1]),(-2.0,-1.0), (-2.0,0.0), (-0.5, 0.0), (-0.5, 1.0), (1.0, 0.5), (2.0, 1.5)]  # CASE 1
   
     # ------------ Path planner ---------------------
    # start = [-0.5, 1.0]
    start = [x0_m[0],x0_m[1]]
    goal = [2,1.5]
    waypoints = planner(start, goal, S_obstacle)
    
    path_x = [p[0] for p in waypoints]
    path_y = [p[1] for p in waypoints]
    start = waypoints[0]
    goal = waypoints[-1]


    # ------------ Path parameterization ---------------------
    path = ParameterizedPath(waypoints, step=0.02)
    sf = float(path.sf) if hasattr(path, "sf") else float(path.s[-1])

    # ------------------- GRAMPC setup ---------------
    Tsim = 100
    plotSteps = 100
    options = "MPC_SPOT.json"  

    # --------------- Cost weights --------------------
    Q = np.array([4, 4, 0, 1, 1, 0.5, 0.5])  # state cost weights  
    R = np.array([[1.0, 1.0, 1.0, 0.05 ]])                     
    ac = 100                                              # collision cost weight
    
    x0 = np.array([x0_m[0], x0_m[1], x0_m[2], 0.0, 0.0, 0.0, 0.0])

    problem = Spot(Q, R, ac, path, S_obstacle, D_obstacle)  
    grampc = Grampc(problem, options, plot_prediction=False)

    grampc.set_param({"x0": x0, "t0": 0.0})

    # Prediction of dynamic obstacle positions over horizon
    th_t = np.linspace(0, grampc.param.Thor, grampc.opt.Nhor)
    problem.dyn_preds = np.zeros_like(th_t)
    problem.t_frac = (grampc.opt.Nhor-1)/grampc.param.Thor

    grampc.estim_penmin(True)
    grampc.print_opts()
    grampc.print_params()

    # Results containers
    vec = GrampcResults(grampc, Tsim, plot_results=True, plot_statistics=True)
    dt = grampc.param.dt

    # For plotting/animation
    trajectory_x, trajectory_y, trajectory_theta = [], [], []
    patho_x, patho_y = [], []
    pred_patho_x, pred_patho_y = [], []

    # Stop conditions
    min_sim_time = 1.0
    pos_threshold = 0.10
    vel_threshold = 0.05
    omega_threshold = 0.05
    s_tol = 1e-1  #1e-4

    min_d_D_obs = min_circ_dist(x0_m[0],x0_m[1],x0_m[2], zobs_0[0], zobs_0[1])-0.3-0.2
    input("Press Enter to start the loop...")
    time.sleep(1)

    # Initialize CSV logging

    csv_filename = f"{name}.csv"
    csv_headers = [
        'index', 'time_t', # Index and time
        'x_s', 'y_s', 'theta_s', 'vlon_s', 'vlat_s', 'omega', 's',  # Current Spot states
        'x_o', 'y_o', 'vx_o', 'vy_o',  # Current obstacle states
        'ex', 'ey', 'etheta', # Errors
        'min_d_D_obs',  # Distances to obstacles
        'alon', 'alat', 'alpha', 'nu',  # Control inputs MPC
        'vlon_spot', 'vlat_spot', 'omega_spot',  # Control inputs Spot
        'J', 'J_ag', #Costs
        'linesearch_step_size',  # Linesearch step size
        'MPC_t',  # MPC time
        'Loop_t', # Loop time
        'Co_t' # To command time
    ]
    log_row = [
        0, 0.0,
        grampc.param.x0[0], grampc.param.x0[1], grampc.param.x0[2], grampc.param.x0[3], grampc.param.x0[4], grampc.param.x0[5], grampc.param.x0[6],
        zobs_0[0], zobs_0[1], 0.0, 0.0,
        0.0, 0.0, 0.0, # Errors
        min_d_D_obs,  # Distances to obstacles
        grampc.sol.unext[0], grampc.sol.unext[1], grampc.sol.unext[2], grampc.sol.unext[3],
        grampc.rws.x[3,3], grampc.rws.x[4,3], grampc.rws.x[5,3],
        0.0, 0.0,
        0.0,  # Linesearch step size
        0.0,  
        0.0, 
        0.0
        ]
            # Write to CSV file
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerow(log_row)
    print(f"Logging data to: {csv_filename}")

    for i, t in enumerate(vec.t):

        print("----- Time step", i, "at t =", t, "-----")
        start_time = time.time_ns()

        # ----------Update dynamic obstacle position----------
        # estimator update: predict to t_abs, take a measurement, update
        K_Obs.predict()
        # z_o = rb.get_rigid_body_data()[OBJECT_ID]  # OptiTrack position of dynamic obstacle
        z_o = np.array([obj_x[i+1], obj_y[i+1]])
        z = np.array([z_o[0], z_o[1]])
        K_Obs.update(z)
        x_obs,vx_obs,y_obs,vy_obs = K_Obs.x

        # Update dynamic obstacle predictions over horizon
        problem.dyn_preds = np.array([CVM(x_obs,vx_obs,y_obs,vy_obs, ti) for ti in th_t])

        # ----------Solve MPC once per step----------
        vec.CPUtime[i] = grampc.run()
        vec.update(grampc, i) 

        # Time to command
        command_time = np.abs(time.time_ns() - start_time)

        # ----Integrate real dynamics with the optimal u over [t, t+dt]---
        sol = solve_ivp(
            grampc.ffct, [t, t + dt], grampc.param.x0,
            args=(grampc.sol.unext, grampc.sol.pnext),
            t_eval=[t + dt]
        )

        # Construct the next state vector
        x_next = sol.y[:, -1]

        grampc.set_param({"x0": x_next, "t0": t + dt})

        # Log for animation
        trajectory_x.append(x_next[0])
        trajectory_y.append(x_next[1])
        trajectory_theta.append(x_next[2])
        
        patho_x.append(z_o[0])
        patho_y.append(z_o[1])

        pred_patho_x.append(x_obs)
        pred_patho_y.append(y_obs)
    
        print(f"Vel lon {x_next[3]}")

        # Termination: reached end-of-path and at rest (matches terminal idea)
        s_now = x_next[6]
        xref, yref, thref = path.ref_at(s_now) 
        pos_errx = abs(x_next[0] - xref)
        pos_erry = abs(x_next[1] - yref)
        pos_errtheta = abs(angle_wrap(x_next[2]-thref))

        # Minimum distance from robot to dynamic obstacle
        min_d_D_obs = min_circ_dist(x_next[0],x_next[1],x_next[2], x_obs, y_obs)-0.3-0.2

        end_time = time.time_ns()
        delta_time = np.abs(end_time-start_time)
        print("Time taken for this iteration: ", (end_time - start_time)*1e-9, " seconds")
        if delta_time*1e-9 < grampc.param.dt:
            time.sleep(grampc.param.dt-delta_time*1e-9)

        # Log data to CSV
        log_row = [
            int(i+1), float(t+dt),
            x_next[0], x_next[1], x_next[2], x_next[3], x_next[4], x_next[5], x_next[6],
            x_obs, y_obs, vx_obs, vy_obs,
            pos_errx, pos_erry, pos_errtheta, # Errors
            min_d_D_obs,  # Distances to dyn obstacle
            grampc.sol.unext[0], grampc.sol.unext[1], grampc.sol.unext[2], grampc.sol.unext[3],
            grampc.rws.x[3,3], grampc.rws.x[4,3], grampc.rws.x[5,3],
            vec.J[i,0], vec.J[i,1],
            float(vec.linesearch_step_size[i]),  # Linesearch step size
            float(vec.CPUtime[i]), 
            delta_time*1e-6, 
            command_time*1e-6
            ]
        
        # Write to CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log_row)

        # if (t > min_sim_time and
        #     s_now >= sf - 1e-3 and
        #     pos_err < pos_threshold and
        #     np.hypot(x_next[3], x_next[4]) < vel_threshold and
        #     abs(x_next[5]) < omega_threshold):
            # if problem.total_cost <= 1e-6:
        if (t > min_sim_time and 
            s_now >= sf and
            pos_errx < pos_threshold and pos_erry < pos_threshold):
                print(f"Reached end-of-path and stopped at t={t:.2f}s, err={np.linalg.norm([pos_errx, pos_erry]):.3f} m")
                break
        if i % plotSteps == 0:
            grampc.plot()
            vec.plot()
            #print("ccost {:.8e}".format(problem.ccost*problem.ac))
            print(f"t={t:.2f}s | pos=({x_next[0]:.2f},{x_next[1]:.2f}) | s={s_now:.2f}/{sf:.2f}")
            print("========================================================")
    time.sleep(1)

    # Animate 
    path_robot = (path_x , path_y)
    trajectory_robot = (trajectory_x, trajectory_y, trajectory_theta)
    path_o = (patho_x, patho_y)
    robot = (problem.rrad, problem.l)
    ppath_o = (pred_patho_x, pred_patho_y)

    animate_spot(robot, start, goal, path_robot, trajectory_robot, S_obstacle, D_obstacle, path_o, ppath_o, dt, name)

    plt.show()

