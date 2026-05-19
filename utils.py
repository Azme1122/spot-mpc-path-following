import numpy as np
import time


def angle_wrap(e):
    # angle wrapped into [-pi, pi]
    # return np.arctan2(np.sin(e), np.cos(e))
    # wrap to [-pi, pi)
    return (e + np.pi) % (2*np.pi) - np.pi

def xdot2vspot(vx, vy, theta):
    vlon = np.cos(theta)*vx + np.sin(theta)*vy
    vlat = -np.sin(theta)*vx + np.cos(theta)*vy
    return vlon, vlat

# ------------- Dynamic obstacle trajectory --------------

def sensor_measure(obsx0, vox, obsy0, voy, t, sigma=0.05, rng=np.random.default_rng(123)):
        return CVM(obsx0, vox, obsy0, voy, t) + rng.normal(0.0, sigma, size=2)

def CVM(x, vx, y, vy, dt):
        x1 = x + vx*dt
        y1 = y + vy*dt
        return x1, y1

def vel_clip(v, vmin=0.05, vstop=1e-6):
    sign = np.sign(v)
    v_abs = np.abs(v)
    if v_abs <= vstop:
        v_clip = 0.0
    elif v_abs < vmin:        
        v_clip = np.clip(v_abs, vmin, None)
    else:
        v_clip = v_abs
    return sign * v_clip

def min_circ_dist(rx,ry,th, obsx, obsy):
    lcos = 0.275 * np.cos(th)
    lsin = 0.275 * np.sin(th)
    c1_pos = np.array([rx - lcos, ry - lsin])  # front circle
    c2_pos = np.array([rx + lcos, ry + lsin])  # rear circle
    obs_pos = np.array([obsx, obsy])
    d1 = np.linalg.norm(c1_pos - obs_pos)
    d2 = np.linalg.norm(c2_pos - obs_pos)
    min_dist = min(d1, d2)

    return min_dist