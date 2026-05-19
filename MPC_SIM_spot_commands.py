import time
import matplotlib.pyplot as plt
import numpy as np
from pygrampc import Grampc, GrampcResults
from Animation_obs import animate_spot
from path_planning import planner
from pos_predictor import K_Spot, K_Obs
from path_param import ParameterizedPath
from mpc_problem import Spot 
from utils import angle_wrap, xdot2vspot, CVM, vel_clip, min_circ_dist
import NNet
from sdk_spot import SDKSpot
import datetime
import csv

SPOT_ID = 39 #6
OBJECT_ID = 7

if __name__ == "__main__":
    # ---------------- Spot and NatNet initialization----------------
    name='CASE3.1'
    rb= NNet.NatNetDataProcessor(server="192.168.1.99", client="192.168.1.20", use_multicast=True)
    rb.start_streaming()

    input(" Press enter to continue... ")

    spot = SDKSpot(hostname="192.168.80.3", username="user", password="c037gcf6n93f")
    spot.connect()
    time.sleep(1)

    x0_m = rb.get_rigid_body_data()[SPOT_ID]
    K_Spot.x = np.array([x0_m[0], 0.0, x0_m[1], 0.0, x0_m[2], 0.0]) #[x,vx,y,vy,theta,omega]
    

    # ---------------- Static obstacles ----------------
    r = 0.30
    rsafe = 0.20
    #S_obstacle = [(-1.19384074, -1.37350976, r, rsafe),(-1.19384074, -1.37350976, r, rsafe),(-1.19384074, -1.37350976, r, rsafe)] 
    #S_obstacle = [(-0.75, 0.13, r, rsafe),(1.15, -1.78, r, rsafe),(1.19384074, 1.37350976, r, rsafe), (-1.16, -1.26, r, rsafe),(-1.8, 0.22, r, rsafe)] #added 12.11
    #S_obstacle = [(-0.75, 0.13, r, rsafe), (-1.16, -1.26, r, rsafe),(-1.8, 0.22, r, rsafe)] #added 12.11
    S_obstacle = [(-1.0, 2.5, r, rsafe), (-0.0, 2.5, r, rsafe),(0.5,1.5, r, rsafe),(0.5,0.5, r, rsafe),(-0.0, 0.0, r, rsafe),(-1.0,0.0, r, rsafe)] #CASE 2 & 3
    #S_obstacle = []  #CASE 1
 

    # --------------- Dynamic obstacle ------------------
    # Initial state estimate for Kalman filter
    zobs_0 = rb.get_rigid_body_data()[OBJECT_ID]  # OptiTrack position
    K_Obs.x = np.array([zobs_0[0], 0.0, zobs_0[1], 0.0])
    rdyn = 0.20
    D_obstacle = [zobs_0[0], zobs_0[1], rdyn, rsafe*1.4]

    # --------------- Waypoints & path ---------------
    #waypoints = [(x0_m[0],x0_m[1]),(-2.0,-1.0), (-2.0,0.0), (-0.5, 0.0), (-0.5, 1.0), (1.0, 0.5), (2.0, 1.5)]  # CASE 1
   
    # ------------ Path planner ---------------------
    #start = [-0.5, 1.0]
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
        'x_s_m', 'y_s_m', 'theta_s_m',  # Optitrack Spot
        'x_o', 'y_o', 'vx_o', 'vy_o',  # Current obstacle states
        'x_o_m', 'y_o_m',  # Optitrack obstacle
        'ex', 'ey', 'etheta', # Errors
        'min_d_S_obs', 'min_d_D_obs',  # Distances to obstacles
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
        x0_m[0], x0_m[1], x0_m[2],
        zobs_0[0], zobs_0[1], 0.0, 0.0,
        zobs_0[0], zobs_0[1],
        0.0, 0.0, 0.0, # Errors
        min_d_D_obs,  # Distances to obstacles
        grampc.rws.u[0,3], grampc.rws.u[1,3], grampc.rws.u[2,3], grampc.rws.u[3,3],
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
        z_o = rb.get_rigid_body_data()[OBJECT_ID]  # OptiTrack position of dynamic obstacle
        z = np.array([z_o[0], z_o[1]])
        K_Obs.update(z)
        x_obs,vx_obs,y_obs,vy_obs = K_Obs.x

        # Update dynamic obstacle predictions over horizon
        problem.dyn_preds = np.array([CVM(x_obs,vx_obs,y_obs,vy_obs, ti) for ti in th_t])

        # ----------Solve MPC once per step----------
        vec.CPUtime[i] = grampc.run()
        vec.update(grampc, i) 

        # ---------------- Apply predicted velocities ----------------
        x_next_p = grampc.rws.x 

        pred_i=int(3)
        vlon_next, vlat_next, omega_next = x_next_p[3][pred_i], x_next_p[4][pred_i], x_next_p[5][pred_i]
        x_next_m, y_next_m, theta_next_m, s_next_m = x_next_p[0][pred_i], x_next_p[1][pred_i], x_next_p[2][pred_i],x_next_p[6][pred_i] 
        
        print("x+s from mpc:", x_next_m, y_next_m, theta_next_m, s_next_m)
        print("v from mpc:", vlon_next, vlat_next, omega_next)

        # Clamp velocities to avoid robot to be stopped.
        if i <= 3:
            vlon_next = vel_clip(vlon_next, vmin=0.1, vstop=1e-6)
            vlat_next = vel_clip(vlat_next, vmin=0.1, vstop=1e-6)
            omega_next = vel_clip(omega_next,vmin=0.02, vstop=1e-4)

        # Send velocity command to Spot
        command_time = np.abs(time.time_ns() - start_time)
        spot.send_velocity_command(vlon_next, vlat_next, omega_next)
              
        
        # ---------------- Measure Spot's states [x,y,theta] ----------------
        # Kalman filter state estimation
        x_measured = rb.get_rigid_body_data()[SPOT_ID]  # OptiTrack position
        print("x_current to kf:", x_measured)
        z_s = np.array([x_measured[0], x_measured[1], x_measured[2]])
        K_Spot.predict()
        z_s[2] = K_Spot.x[4] + angle_wrap(z_s[2] - K_Spot.x[4])
        K_Spot.update(z_s) 
        x_s,vx_s,y_s,vy_s,theta_s,omega_s = K_Spot.x
        theta_s = angle_wrap(theta_s) #angle wrapping
        print("theta_kf, omega_kf: ",theta_s," ", omega_s)
        vlon, vlat = xdot2vspot(vx_s, vy_s, theta_s)

        # Construct the next state vector
        x_next = np.array([x_s, y_s, theta_s, vlon, vlat,omega_s, x_next_p[6][1]])
        print("xnext to mpc", x_next)

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
        

        print("s_now, sf: ", s_now," ", sf)
        print("pos_errx: ", pos_errx, " posth", pos_threshold)
        print("pos_erry: ", pos_erry, " posth", pos_threshold)
        print("t: ", t)
        print("u: ", grampc.rws.u[0][1],grampc.rws.u[1][1],grampc.rws.u[2][1])

        end_time = time.time_ns()
        delta_time = np.abs(end_time-start_time)
        print("Time taken for this iteration: ", (end_time - start_time)*1e-9, " seconds")
        if delta_time*1e-9 < grampc.param.dt:
            time.sleep(grampc.param.dt-delta_time*1e-9)

        # Log data to CSV
        log_row = [
            int(i+1), float(t+dt),
            x_next[0], x_next[1], x_next[2], x_next[3], x_next[4], x_next[5], x_next[6],
            x_measured[0], x_measured[1], x_measured[2],
            x_obs, y_obs, vx_obs, vy_obs,
            z_o[0], z_o[1],
            pos_errx, pos_erry, pos_errtheta, # Errors
            min_d_D_obs,  # Distances to dyn obstacle
            grampc.rws.u[0,3], grampc.rws.u[1,3], grampc.rws.u[2,3], grampc.rws.u[3,3],
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


        if (t > min_sim_time and
            s_now >= sf and
            pos_errx < pos_threshold and pos_erry < pos_threshold):
            print(f"Reached end-of-path and stopped at t={t:.2f}s, err={pos_errx:.3f} m")
            break
        if i % plotSteps == 0:
            grampc.plot()
            vec.plot()
            print(f"t={t:.2f}s | pos=({x_next[0]:.2f},{x_next[1]:.2f}) | s={s_now:.2f}/{sf:.2f}")
            print("========================================================")
        #time.sleep(2)
    time.sleep(1)
    spot.stop_and_disconnect()
    # Animate 
    path_robot = (path_x , path_y)
    trajectory_robot = (trajectory_x, trajectory_y, trajectory_theta)
    path_o = (patho_x, patho_y)
    robot = (problem.rrad, problem.l)
    ppath_o = (pred_patho_x, pred_patho_y)

    animate_spot(robot, start, goal, path_robot, trajectory_robot, S_obstacle, D_obstacle, path_o, ppath_o, dt,name)
    plt.show()


