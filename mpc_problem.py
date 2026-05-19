from pygrampc import ProblemDescription
from utils import angle_wrap
import numpy as np


class Spot(ProblemDescription):
    """
    State x = [px, py, theta, vx, vy, omega, s]
    Input u = [ax, ay, alpha, nu]   # nu = \dot{s} (virtual speed along path)
    Q = [Qx, Qy, Qth, Qvx, Qvy, Qom, Qs]
    R = [Rax, Ray, Ralp, Rnu]
    obstacles
    """
    def __init__(self, Q, R, ac, path, S_obstacles=None, D_obstacle=None):
        super().__init__()
        self.Ng   = 0
        self.NgT  = 0            # no terminal equality constraints
        self.Nh   = 6 + 2 * len(S_obstacles)           # vlon x2, vlat x2, omega x2 limits + 2 circles (robot) x 4 static obstacles
        self.NhT  = 2            # 2 terminal inequality constraints (x,y , theta errors)

        self.Nx   = 7
        self.Nu   = 4            # extra control nu
        self.Np   = 0            # no external parameters needed

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.ac = ac

        self.path = path
        self.sf = float(path.s_grid[-1])      # total arc length

        self.delta_phi = 1e-3

        # spot parameters
        self.l = 0.275    #0.55/2     # spot's length/2 
        self.rrad = 0.4    # radius of the spot's circles 

        
        self.lcos = None
        self.lsin = None
        self.c1_pos = None
        self.c2_pos = None

        # obstacles
        self.obstacles = S_obstacles
        self.rDO = D_obstacle[2]
        self.rsafe = D_obstacle[3]  

        self.t_frac = None
        self.dyn_preds = None
        self.pred_xDO = 0.0
        self.pred_yDO = 0.0
        self.t0_abs = 0.0 

        # collision cost
        self.ccost = 0.0
        self.dccost_dx0 = 0.0
        self.dccost_dx1 = 0.0
        self.dccost_dx2 = 0.0
        self.epsc = 0.2

        self.QhT = 1.0
        self.rT = 0.5
        self.rTthT = 0.1



    # ---------- helpers ----------
    # @staticmethod
    # def _wrap_angle(e):
    #     # angle wrapped into [-pi, pi]
    #     return np.arctan2(np.sin(e), np.cos(e))

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
        
    def circle_pos(self, x):
        angle = angle_wrap(x[2])
        self.lcos = self.l * np.cos(angle)
        self.lsin = self.l * np.sin(angle)
        self.c1_pos = np.array([x[0] + self.lcos, x[1] + self.lsin])  # front circle
        self.c2_pos = np.array([x[0] - self.lcos, x[1] - self.lsin])  # rear circle
        
        
        

    # ---------- dynamics z_dot = f(z,u) ----------
    def ffct(self, out, t, x, u, p):
        c = np.cos(x[2]); s = np.sin(x[2])
        out[0] = x[3]*c - x[4]*s           # xdot
        out[1] = x[3]*s + x[4]*c           # ydot
        out[2] = x[5]                      # thetadot
        out[3] = u[0]                      # vxdot
        out[4] = u[1]                      # vydot
        out[5] = u[2]                      # omegadot
        out[6] = u[3]                      # sdot = nu  

    # Jacobian-vector products
    def dfdx_vec(self, out, t, x, vec, u, p):
        c = np.cos(x[2]); s = np.sin(x[2])

        out[0] = 0.0                                   # ∂f/∂x has no x-column
        out[1] = 0.0                                   # no y-column
        out[2] = (-x[3]*s - x[4]*c) * vec[0] + (x[3]*c - x[4]*s) * vec[1]   # θ-column
        out[3] =              c * vec[0] +             s * vec[1]           # vx-column
        out[4] =             -s * vec[0] +             c * vec[1]           # vy-column
        out[5] = vec[2]                                                     # ω-column
        out[6] = 0.0                                                        # s-column
                            # sdot has no state dependence

    def dfdu_vec(self, out, t, x, vec, u, p):
        # out has length Nu=4; vec has length Nx=7
        out[0] = vec[3]   # from dvx/ dax = 1
        out[1] = vec[4]   # from dvy/ day = 1
        out[2] = vec[5]   # from domega/ dα = 1
        out[3] = vec[6]   # from ds/ dν = 1

    # ---------- dynamic obstacle collision cost ----------------
    def collision(self, x, t_rel):
        self.circle_pos(x) # spot circles positions
        c_pos = np.array([self.c1_pos, self.c2_pos])
        rtotal = self.rrad + self.rDO + self.rsafe
        
        self.ccost = 0.0
        # print(index)
        do_pos = self.dyn_preds[int(t_rel*self.t_frac)]
        # pred = self.dyn_preds
        # # for pred in self.dyn_preds:
        # pred_val = pred(self.t0_abs + t_rel) ### Correct this part.
        # do_pos = pred_val[:2]  
        
        # Distance calculation for both circles wrt dynamic obstacle
        dist_vectors = do_pos - c_pos  # Shape: (2,2)
        d_squared = np.sum(dist_vectors**2, axis=1)  
        #d = np.sqrt(d_squared)
        
        # Collision condition check
        sd = d_squared - rtotal**2
        h = self.epsc - sd
        
        # Only compute cost where h > 0 (collision region)
        collision = h > 0
        if np.any(collision):
            self.ccost += np.sum(h[collision])

        return self.ccost
    
    # ---------- dynamic obstacle collision cost gradients ----------------
    def collision_g(self, x, t_rel):
        self.circle_pos(x) #spot circles positions
        c_pos = np.array([self.c1_pos, self.c2_pos])
        rtotal = self.rrad + self.rDO + self.rsafe
        
        # Initialize gradients
        self.dccost_dx0 = 0.0
        self.dccost_dx1 = 0.0 
        self.dccost_dx2 = 0.0
        
        # Pre-compute theta factors for circle position derivatives
        theta_factors = np.array([
            [-self.lsin, self.lcos],   # circle 1: dc1/dtheta coefficients
            [self.lsin, -self.lcos]    # circle 2: dc2/dtheta coefficients
        ])

        do_pos = self.dyn_preds[int(t_rel*self.t_frac)]
        # pred=self.dyn_preds
        # # for pred in self.dyn_preds:
        # pred_val = pred(self.t0_abs + t_rel)
        # do_pos = pred_val[:2]  # Direct slice
        
        # Match collision function exactly: use squared distances
        dist_vectors = do_pos - c_pos  # Shape: (2,2) 
        d_squared = np.sum(dist_vectors**2, axis=1)  # Same as collision function
        
        # Collision conditions (match collision function)
        sd = d_squared - rtotal**2  # Same formulation as collision function
        h = self.epsc - sd          # h = epsc - (d² - r²)
        collision = h > 0

        # if not np.any(collision):
        #     continue  # Skip if no active collisions
        
        # Only compute gradients for active collisions
        h_active = h[collision]
        dist_vectors_active = dist_vectors[collision]
        
        # Gradient computation for h = epsc - (d² - r²)
        
        # Position gradients: dh/dx0 = 2*(x_obs - x_circle), dh/dx1 = 2*(y_obs - y_circle)  
        self.dccost_dx0 += np.sum(2.0 * dist_vectors_active[:, 0])
        self.dccost_dx1 += np.sum(2.0 * dist_vectors_active[:, 1])
        
        # Theta gradients: dh/dtheta = 2*[(x_obs-x_c)*dc_x/dtheta + (y_obs-y_c)*dc_y/dtheta]
        theta_contrib_active = np.sum(dist_vectors_active * theta_factors[collision], axis=1)
        self.dccost_dx2 += np.sum(2.0 * theta_contrib_active)

        return self.dccost_dx0, self.dccost_dx1, self.dccost_dx2

    # ---------- running cost l(x,u) ----------
    def lfct(self, out, t, x, u, p, xdes, udes):
        # path reference at current s
        xref, yref, thref = self.path.ref_at(x[6])
        ex = x[0] - xref
        ey = x[1] - yref
        eth = angle_wrap(x[2] - thref)

        # position + heading to path, small velocity + omega damping, s-progress term
        pos_cost = self.Q[0]*ex*ex + self.Q[1]*ey*ey
        th_cost  = self.Q[2]*eth*eth  
        vel_cost = self.Q[3]*x[3]*x[3] + self.Q[4]*x[4]*x[4] + self.Q[5]*x[5]*x[5]
        s_cost   = self.Q[6]*self._phi(x[6])

        du = u - udes
        u_cost = (self.R * du*du).sum()

        # dynamic obstacle collision cost 
        self.ccost = self.collision(x,t)
        collision_cost = self.ac * self.ccost

        self.total_cost = pos_cost + th_cost + vel_cost + s_cost + u_cost + collision_cost

        out[0] = self.total_cost

    def dldx(self, out, t, x, u, p, xdes, udes):
        # --- fetch path reference at current s ---
        self.dccost_dx0, self.dccost_dx1, self.dccost_dx2 = self.collision_g(x,t)

        xref, yref, thref = self.path.ref_at(x[6])

        # errors
        ex  = x[0] - xref
        ey  = x[1] - yref
        eth = angle_wrap(x[2] - thref)    # or angle_diff(x[2], thref)

        # tangent wrt s (since the path is parameterized by arc length)
        tx = np.cos(thref)
        ty = np.sin(thref)

        # unpack weights (Q = [Qx, Qy, Qtheta, Qvx, Qvy, Qomega, Qs])
        Qx, Qy, Qth, Qvx, Qvy, Qom, Qs = self.Q


        # partials wrt states
        out[0] = 2.0 * Qx  * ex   +   self.ac * self.dccost_dx0     # dJ/dx
        out[1] = 2.0 * Qy  * ey   +   self.ac * self.dccost_dx1     # dJ/dy
        out[2] = 2.0 * Qth * eth  +   self.ac * self.dccost_dx2     # dJ/dtheta 
        out[3] = 2.0 * Qvx * x[3]                                   # dJ/dvlon
        out[4] = 2.0 * Qvy * x[4]                                   # dJ/dvlat
        out[5] = 2.0 * Qom * x[5]                                   # dJ/domega
        out[6] = Qs * self._dphi(x[6]) - 2.0 * Qx * ex * tx - 2.0 * Qy * ey * ty

        

    def dldu(self, out, t, x, u, p, xdes, udes):
        du = u - udes
        out[:] = 2*self.R*du

    def hfct(self, out, t, x, u, p):
        self.circle_pos(x)

        max_vlon = 1.6
        max_vlat = 0.6
        max_omega = 1.5

        out[0] = x[3] - max_vlon         # vlon <= max_vlon
        out[1] = -x[3] - max_vlon        # -vlon <= max_vlon  -> vx >= -max_vlon

        out[2] = x[4] - max_vlat         # vlat <= max_vlat
        out[3] = -x[4] - max_vlat        # vlat >= -max_vlat

        out[4] = x[5] - max_omega        # omega <= max_omega
        out[5] = -x[5] - max_omega       # omega >= -max_omega

        #---------------static obstacles constraints--------------
        for i, (ox, oy, orad, rsafe) in enumerate(self.obstacles): 
            obs_pos = np.array([ox, oy])
            dc1 = np.linalg.norm(self.c1_pos - obs_pos)
            dc2 = np.linalg.norm(self.c2_pos - obs_pos)
            rtotal = orad + self.rrad + rsafe

            #1 collision when: distance_c1**2 - r**2 <= 0 - first circle
            out[6 + i] = -(dc1**2 - rtotal**2)   #6, 7 , 8 constraint

            #2 collision when: distance_c2**2 - r**2 <= 0 - second circle
            out[6 + i + len(self.obstacles)] = -(dc2**2 - rtotal**2)   #9, 10 , 11 constraint

    def dhdx_vec(self, out, t, x, u, p, vec):
        # grad(h)*vec
        self.circle_pos(x)
        out[:] = 0.0
        out[3] = vec[0] - vec[1]
        out[4] = vec[2] - vec[3]
        out[5] = vec[4] - vec[5]

        sin_theta = np.sin(x[2])
        cos_theta = np.cos(x[2])
        

        n_obs = len(self.obstacles)
        for i, (ox, oy, orad, rsafe) in enumerate(self.obstacles):
            #position differences
            dx = x[0] - ox
            dy = x[1] - oy
            
            # Circle 1 gradients: h = -(dc1² - rtotal²)
            grad_xc1 = -2 * (dx + self.lcos) # dh/dx0
            grad_yc1 = -2 * (dy + self.lsin) # dh/dx1
            grad_thetac1 = 2 * self.l * (dx * sin_theta - dy * cos_theta) # dh/dx2

            # Circle 2 gradients: h = -(dc2² - rtotal²) 
            grad_xc2 = -2 * (dx - self.lcos)
            grad_yc2 = -2 * (dy - self.lsin)
            grad_thetac2 = -grad_thetac1  

            # Accumulative contributions
            vec_c1 = vec[6 + i]
            vec_c2 = vec[6 + i + n_obs]
            
            out[0] += vec_c1 * grad_xc1 + vec_c2 * grad_xc2
            out[1] += vec_c1 * grad_yc1 + vec_c2 * grad_yc2
            out[2] += vec_c1 * grad_thetac1 + vec_c2 * grad_thetac2


    def dhdu_vec(self, out, t, x, u, p, vec):
        out[:] = 0.0

    def hTfct(self, out, T, x, u):
        xref, yref, thref = self.path.ref_at(x[6])
        ex = x[0] - xref
        ey = x[1] - yref
        eth = angle_wrap(x[2] - thref)


        ezx = self.QhT*ex*ex
        ezy = self.QhT*ey*ey
        ezth  = self.QhT*eth*eth

        out[0] = ezx + ezy - self.rT
        out[1] = ezth - self.rTthT

        #out[2] = ezth - rTth


    def dhTdx_vec(self, out, T, x, u, vec):
        xref, yref, thref = self.path.ref_at(x[6])
        ex = x[0] - xref
        ey = x[1] - yref
        eth = angle_wrap(x[2] - thref)

        Q = 1

        tx = np.cos(thref)  # dx_ref/ds
        ty = np.sin(thref)  # dy_ref/ds

        out[:] = 0.0
        out[0] = 2*Q*ex*vec[0]
        out[1] = 2*Q*ey*vec[0]
        out[2] = 2*Q*eth*vec[1]
        out[6] = -2*Q*ex*tx*vec[0] - 2*Q*ey*ty*vec[0]
