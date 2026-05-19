import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# --------------Global constants -----------------
DT = 0.1  # Time step

# -------------- Kalman Filter SPOT --------------
# Constants
## Q matrix values (process noise covariance) (From variable data)
VAR_AX_S = 13.122166287927387  # m/s^2
VAR_AY_S = 16.15854932536918   # m/s^2
VAR_Aθ_S = np.deg2rad(100)  # rad/s^2

## R matrix values (measurement noise covariance) (From constant data)
VAR_X_S = 2.33646958e-01 # m
VAR_Y_S = 2.55554676e-01 # m
VAR_θ_S = np.deg2rad(1.0e-1)  # rad

## P matrix values (initial estimate covariance) (From variable data)
VAR_XP_S = 0.23364695765305632
VAR_DX_P_S = 0.026506626686528083
VAR_YP_S = 0.2555546758757689
VAR_DY_P_S = 0.024398979684867744
VAR_θP_S = np.deg2rad(1.0e-1)
VAR_Dθ_P_S = np.deg2rad(0.0015408552528218648)

# Class object K_SPOT->States: [x, vx, y, vy, theta, omega]
K_Spot = KalmanFilter(dim_x=6, dim_z=3)

# State transition matrix F
K_Spot.F = np.array([[1, DT, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, DT, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, DT],
                     [0, 0, 0, 0, 0, 1]], dtype=float)

# Process noise covariance Q
qx_s = Q_discrete_white_noise(dim=2, dt=DT, var=VAR_AX_S)     # for [x, vx]
qy_s = Q_discrete_white_noise(dim=2, dt=DT, var=VAR_AY_S)     # for [y, vy]
qt_s = Q_discrete_white_noise(dim=2, dt=DT, var=VAR_Aθ_S)     # for [theta, omega]
K_Spot.Q = block_diag(qx_s, qy_s, qt_s)

# Measurement matrix H
K_Spot.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0]], dtype=float)

# Measurement noise covariance R
K_Spot.R = np.diag([VAR_X_S, VAR_Y_S, VAR_θ_S])

# Initial state estimate
K_Spot.x = np.array([0, 0, 0, 0, 0, 0])

# Initial estimate covariance P
K_Spot.P = np.diag([VAR_XP_S, VAR_DX_P_S, VAR_YP_S, VAR_DY_P_S, VAR_θP_S, VAR_Dθ_P_S])



# -------------- Kalman Filter OBSTACLE --------------
# Constants
## Q matrix values (process noise covariance)
VAR_AX_O = 19.139744758281797  # m/s^2
VAR_AY_O = 21.91594210430449   # m/s^2

## R matrix values (measurement noise covariance)
VAR_X_O = 0.20034521 # m
VAR_Y_O = 0.08932684 # m

## P matrix values (initial estimate covariance)
VAR_XP_O = 1.7579805116474108
VAR_DX_P_O = 0.20981335130261528
VAR_YP_O = 0.6335995929409879
VAR_DY_P_O = 0.24002221422927952

# Class object K_SPOT->States: [x, vx, y, vy]
K_Obs = KalmanFilter(dim_x=4, dim_z=2)

# State transition matrix F
K_Obs.F = np.array([[1, DT, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, DT],
                    [0, 0, 0, 1]], dtype=float)

# Process noise covariance Q
qx_o = Q_discrete_white_noise(dim=2, dt=DT, var=VAR_AX_O)     # for [x, vx]
qy_o = Q_discrete_white_noise(dim=2, dt=DT, var=VAR_AY_O)     # for [y, vy]
K_Obs.Q = block_diag(qx_o, qy_o)

# Measurement matrix H
K_Obs.H = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0]], dtype=float)

# Measurement noise covariance R
K_Obs.R = np.diag([VAR_X_O, VAR_Y_O])

# Initial state estimate
K_Obs.x = np.array([0, 0, 0, 0])

# Initial estimate covariance P
K_Obs.P = np.diag([VAR_XP_O, VAR_DX_P_O, VAR_YP_O, VAR_DY_P_O])

import numpy as np

class CVKF2D:
        def __init__(self, x0, P0=None, q=0.2, r=0.05):
            self.x = np.asarray(x0, float)  # [x y vx vy]
            self.P = np.eye(4) * 1.0 if P0 is None else P0.copy()
            self.q = float(q)                
            self.R = np.eye(2) * (r**2)      
            self.last_t = None

        def _F_Q(self, dt):
            F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1,  0],
                          [0, 0, 0,  1]], dtype=float)
            dt2 = dt*dt; dt3 = dt2*dt
            q = self.q
            Q = q * np.array([[dt3/3,    0,    dt2/2, 0],
                              [0,     dt3/3,    0,    dt2/2],
                              [dt2/2,    0,      dt,  0],
                              [0,      dt2/2,    0,    dt]], dtype=float)
            return F, Q

        def predict_to(self, t):
            if self.last_t is None:
                self.last_t = t
                # return self.x
            dt = max(1e-3, float(t - self.last_t))
            F, Q = self._F_Q(dt)
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q
            self.last_t = t
            # return self.x

        def update(self, z):
            H = np.array([[1,0,0,0],
                          [0,1,0,0]], dtype=float)
            y  = np.asarray(z, float) - H @ self.x
            S  = H @ self.P @ H.T + self.R
            K  = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ H) @ self.P
            return self.x

        def predict_position_at(self, t_future):
            # roll a copy fwd; do not change internal state
            if self.last_t is None:
                return self.x[:2].copy()
            dt = max(0.0, float(t_future - self.last_t))
            # print("=================last_t===================", self.last_t)
            F, _ = self._F_Q(dt)
            xf = F @ self.x
            return xf[:2].copy()
