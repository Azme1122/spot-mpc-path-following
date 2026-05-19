import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from pygrampc import ProblemDescription, Grampc, GrampcResults
from WO_animation import animate_spot
# import matplotlib
# matplotlib.use('tkagg')  # Add this line
class ParameterizedPath:
 
    def __init__(self, waypoints, step: float = 0.05):
        wp = np.asarray(waypoints, dtype=float)   #Converts list of tuples → NumPy array
        if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:   #check: 2D array,Each row has 2 values (x, y),At least 2 points
            raise ValueError("waypoints must be shape (M,2) with M>=2")
        self.path = self._densify(wp, step=step)              
        diffs = np.diff(self.path, axis=0)                    
        segL = np.linalg.norm(diffs, axis=1)                  
        self.s_grid = np.concatenate(([0.0], np.cumsum(segL)))
        self.sf = float(self.s_grid[-1])

        theta = np.arctan2(diffs[:, 1], diffs[:, 0])          
        self.theta_grid = np.concatenate((theta, [theta[-1]]))

    @staticmethod
    def _densify(wp: np.ndarray, step: float) -> np.ndarray: #takes two arguments:a NumPy array,a float and a return a NumPy array
        """Linear interpolation along each segment so no gap exceeds `step`."""
        dense = [wp[0]]
        for i in range(len(wp) - 1):
            a, b = wp[i], wp[i + 1]
            seg = b - a
            L = float(np.linalg.norm(seg))
            if L < 1e-12:  
                continue  # skip zero-length hops
            n = max(2, int(np.ceil(L / step)))  # include end point,ceil() of 6.3 is 7
            ts = np.linspace(0.0, 1.0, n, dtype=float) 
            pts = (1.0 - ts)[:, None] * a + ts[:, None] * b
            dense.extend(pts[1:])  # avoid duplicating 'a'
        return np.asarray(dense, dtype=float)

    def ref_at(self, s: float):
        """Return (x_ref, y_ref, theta_ref) at arc-length s (clamped to [0, sf])."""
        if self.sf <= 0.0:
            x, y = self.path[0]
            th = self.theta_grid[0]
            return float(x), float(y), float(th)

        s = float(np.clip(s, 0.0, self.sf))
        # locate segment index i with s in [s_i, s_{i+1}]
        i = int(np.searchsorted(self.s_grid, s, side="right") - 1) 
        i = max(0, min(i, len(self.s_grid) - 2)) #Makes sure i is a valid index (not out of range).
        s0, s1 = self.s_grid[i], self.s_grid[i + 1]
        a, b = self.path[i], self.path[i + 1]
        r = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        pos = (1.0 - r) * a + r * b
        th = self.theta_grid[i]
        return float(pos[0]), float(pos[1]), float(th)


# Step2: MPC problem (7D state incl. s, 4D input incl. nu)
class WithoutCollision07(ProblemDescription):
    """
    State x = [px, py, theta, vx, vy, omega, s]
    Input u = [ax, ay, alpha, nu]   # nu = \dot{s} (virtual speed along path)
    Q = [Qx, Qy, Qth, Qvx, Qvy, Qom, Qs]
    R = [Rax, Ray, Ralp, Rnu]
    """
    def __init__(self, Q, R, path):
        super().__init__()
        self.Ng   = 0
        self.NgT  = 6              # x,y,theta on path + vx=vy=omega=0 at horizon end
        self.Nh   = 4              # |vx|, |vy| limits
        self.NhT  = 0

        self.Nx   = 7
        self.Nu   = 4              #  extra control nu
        self.Np   = 0              # no external parameters needed

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.path = path
        self.sf = float(path.s_grid[-1])      # total arc length

        self.delta_phi = 1e-3

    
    @staticmethod
    def _wrap_angle(e):
        # angle wrapped into [-pi, pi]
        return np.arctan2(np.sin(e), np.cos(e))

    def _phi(self, s):
        # smoothed absolute ϕ(s, sf)
        d = s - self.sf
        ad = abs(d)
        if ad > self.delta_phi:
            return ad - 0.5*self.delta_phi
        else:
            return 0.5*(d*d)/self.delta_phi

    def _dphi(self, s):
        # derivative of ϕ wrt s
        d = s - self.sf
        ad = abs(d)
        if ad > self.delta_phi:
            return np.sign(d)
        else:
            return d/self.delta_phi

    # ---------- dynamics z_dot = f(z,u) ----------
    def ffct(self, out, t, x, u, p):
        c = np.cos(x[2]); s = np.sin(x[2])
        out[0] = x[3]*c - x[4]*s           # xdot
        out[1] = x[3]*s + x[4]*c           # ydot
        out[2] = x[5]                      # thetadot
        out[3] = u[0]                      # vxdot
        out[4] = u[1]                      # vydot
        out[5] = u[2]                      # omegadot
        out[6] = u[3]                      # sdot = nu   <-- NEW

    # Jacobian-vector products
    def dfdx_vec(self, out, t, x, vec, u, p):
        c = np.cos(x[2]); s = np.sin(x[2])

        out[0] = 0.0                                   # ∂f/∂x has no x-column
        out[1] = 0.0                                   # no y-column
        out[2] = (-x[3]*s - x[4]*c) * vec[0] + (x[3]*c - x[4]*s) * vec[1]   # θ-column
        out[3] =              c * vec[0] +             s * vec[1]           # vx-column
        out[4] =             -s * vec[0] +             c * vec[1]           # vy-column
        out[5] = vec[2]                                                     # ω-column
        out[6] = 0.0                                                         # s-column
                            # sdot has no state dependence

    def dfdu_vec(self, out, t, x, vec, u, p):
        # out has length Nu=4; vec has length Nx=7
        out[0] = vec[3]   # from dvx/ dax = 1
        out[1] = vec[4]   # from dvy/ day = 1
        out[2] = vec[5]   # from domega/ dα = 1
        out[3] = vec[6]   # from ds/ dν = 1



    # ---------- running cost l(x,u) ----------
    def lfct(self, out, t, x, u, p, xdes, udes):
        # path reference at current s
        xref, yref, thref = self.path.ref_at(x[6])
        ex = x[0] - xref
        ey = x[1] - yref
        eth = self._wrap_angle(x[2] - thref)

        # position + heading to path, small velocity + omega damping, s-progress term
        pos_cost = self.Q[0]*ex*ex + self.Q[1]*ey*ey
        th_cost  = self.Q[2]*eth*eth
        # vel_cost = self.Q[3]*x[3]*x[3] + self.Q[4]*x[4]*x[4] + self.Q[5]*x[5]*x[5]
        s_cost   = self.Q[6]*self._phi(x[6])

      
        du = u - udes
        u_cost  = (self.R * du*du).sum()

        out[0] = pos_cost + th_cost + s_cost + u_cost #+ #vel_cost

    def dldx(self, out, t, x, u, p, xdes, udes):
        # --- fetch path reference at current s ---
    
        xref, yref, thref = self.path.ref_at(x[6])
        

        # errors
        ex  = x[0] - xref
        ey  = x[1] - yref
        eth = self._wrap_angle(x[2] - thref)    # or angle_diff(x[2], thref)

        # tangent wrt s (since the path is parameterized by arc length)
        tx = np.cos(thref)
        ty = np.sin(thref)

        # unpack weights (Q = [Qx, Qy, Qtheta, Qvx, Qvy, Qomega, Qs])
        Qx, Qy, Qth, Qvx, Qvy, Qom, Qs = self.Q

        # partials wrt states
        out[0] = 2.0 * Qx  * ex                  # dJ/dx
        out[1] = 2.0 * Qy  * ey                  # dJ/dy
        out[2] = 2.0 * Qth * eth                 # dJ/dtheta
        out[3] = 2.0 * Qvx * x[3]                # dJ/dvx
        out[4] = 2.0 * Qvy * x[4]                # dJ/dvy
        out[5] = 2.0 * Qom * x[5]                # dJ/domega
        out[6] = Qs * self._dphi(x[6]) - 2.0 * Qx * ex * tx - 2.0 * Qy * ey * ty

    def dldu(self, out, t, x, u, p, xdes, udes):
        du = u - udes
        out[:] = 2*self.R*du

    # ---------- simple speed inequalities on vx, vy ----------
    def hfct(self, out, t, x, u, p):
        vlim = 1.6
        out[0] = x[3] - vlim         # vx <= vlim
        out[1] = -x[3] - vlim        # -vx <= vlim  -> vx >= -vlim
        out[2] = x[4] - vlim         # vy <= vlim
        out[3] = -x[4] - vlim        # vy >= -vlim

    def dhdx_vec(self, out, t, x, u, p, vec):
        # grad(h)*vec
        out[:] = 0.0
        out[3] = vec[0] - vec[1]
        out[4] = vec[2] - vec[3]

    def dhdu_vec(self, out, t, x, u, p, vec):
        out[:] = 0.0

    # ---------- terminal equalities: stop-on-path ----------
    def gfct_T(self, out, t, x, u, p, xdes):
        xref, yref, thref = self.path.ref_at(x[6])
        out[0] = x[0] - xref           # x(T) on path at s(T)
        out[1] = x[1] - yref           # y(T) on path at s(T)
        out[2] = self._wrap_angle(x[2] - thref)  # theta(T) = theta_ref(s(T))
        out[3] = x[3]                  # vx(T) = 0
        out[4] = x[4]                  # vy(T) = 0
        out[5] = x[5]                  # omega(T) = 0

    def dgdxT_vec(self, out, t, x, u, p, vec):
        # grad(g_T)*vec  for g = [x-xref(s), y-yref(s), th-thref(s), vx, vy, omega]
        xref, yref, thref = self.path.ref_at(x[6])
        tx = np.cos(thref); ty = np.sin(thref)   # dx_ref/ds, dy_ref/ds
        
        out[0] = vec[0]               
        out[1] = vec[1]             
        out[2] = vec[2]              
        out[3] = vec[3]               
        out[4] = vec[4]              
        out[5] = vec[5]               
        out[6] = -tx*vec[0] - ty*vec[1]



# =========================
# Step3: Main function
# =========================
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    # --- Waypoints & path ---
    waypoints = [(0, 0), (1, 5), (10, 5)]
    path_x = [p[0] for p in waypoints]
    path_y = [p[1] for p in waypoints]

    # dense polyline + arc-length grids (use your corrected Step 01 class)
    path = ParameterizedPath(waypoints, step=0.02)
    sf = float(path.sf) if hasattr(path, "sf") else float(path.s[-1])

    # --- GRAMPC setup ---
    Tsim = 50.0
    plotSteps = 100
    options = "C:\\Users\\ivale\\Documents\\FAU\\spot_path_following_control\\MPC_SPOT.json"  # your json is fine


    Q = np.array([10.0, 10.0, 1.5, 0.0, 0.0, 0.0, .4])  # 7 entries
    R = np.array([1.0, 1.0, 0.5, 0.05])                # 4 entries (ax, ay, alpha, nu)
    problem = WithoutCollision07(Q, R, path)  # <-- your corrected Step 02 class (Nu=4, Nx=7, NgT=6)
    grampc = Grampc(problem, options, plot_prediction=False)
    grampc.set_param({"Thor": 2.0})

    umax = np.array([3.0, 3.0, 2.0, 1.6])   # ax, ay, alpha, nu_max ≈ vmax
    umin = np.array([-3.0, -3.0, -2.0, 0.0])
    grampc.set_param({"umax": umax, "umin": umin})

    grampc.estim_penmin(True)
    grampc.print_opts()
    grampc.print_params()

    # Results containers
    vec = GrampcResults(grampc, Tsim, plot_results=True, plot_statistics=True)
    dt = grampc.param.dt

    # Initial state: [x, y, theta, vx, vy, omega, s]
    # Start exactly at first waypoint and s=0
    x0 = np.array([waypoints[0][0], waypoints[0][1], 0.0, 0.0, 0.0, 0.0, 0.0])
    grampc.set_param({"x0": x0, "t0": 0.0})

    # Desired input center (for ||u-udes||_R^2); zero is fine
    grampc.set_param({"udes": np.zeros(4)})

    # For plotting/animation
    trajectory_x, trajectory_y, trajectory_theta = [], [], []

    # Stop conditions
    min_sim_time = 1.0
    pos_threshold = 0.005
    vel_threshold = 0.05
    omega_threshold = 0.05
    s_tol = 1e-4

    data_x=np.copy(np.append(x0,0.0))

    for i, t in enumerate(vec.t):
        # Solve MPC once per step
        vec.CPUtime[i] = grampc.run()
        vec.update(grampc, i)

        # Integrate real dynamics with the optimal u over [t, t+dt]
        sol = solve_ivp(
            grampc.ffct, [t, t + dt], grampc.param.x0,
            args=(grampc.sol.unext, grampc.sol.pnext),
            t_eval=[t + dt]
        )
        x_next = sol.y[:, -1]
        data_x=np.vstack((data_x,np.append(x_next,t+dt)))

        # Warm start next step
        grampc.set_param({"x0": x_next, "t0": t + dt})

        # Log for animation
        trajectory_x.append(x_next[0])
        trajectory_y.append(x_next[1])
        trajectory_theta.append(x_next[2])

        # Termination: reached end-of-path and at rest (matches terminal idea)
        s_now = x_next[6]
        xr, yr, th_ref = path.ref_at(s_now)
        pos_err = np.hypot(x_next[0] - xr, x_next[1] - yr)

        if (t > min_sim_time and
            s_now >= sf - 1e-3 and
            pos_err < pos_threshold and
            np.hypot(x_next[3], x_next[4]) < vel_threshold and
            abs(x_next[5]) < omega_threshold):
            print(f"Reached end-of-path and stopped at t={t:.2f}s, err={pos_err:.3f} m")
            break

        if i % plotSteps == 0:
            grampc.plot()
            vec.plot()
            print(f"t={t:.2f}s | pos=({x_next[0]:.2f},{x_next[1]:.2f}) | s={s_now:.2f}/{sf:.2f}")

    # Animate (your helper)
    # animate_spot(path_x, path_y, trajectory_x, trajectory_y, trajectory_theta, dt)
    plt.show()  
    # np.save("MPC_SPOT_data.npy", data_x)