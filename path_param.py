import numpy as np
class ParameterizedPath:
 
    def __init__(self, waypoints, step: float = 0.05):
        wp = np.asarray(waypoints, dtype=float)
        if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
            raise ValueError("waypoints must be shape (M,2) with M>=2")
        self.path = self._densify(wp, step=step)             
        diffs = np.diff(self.path, axis=0)                    
        segL = np.linalg.norm(diffs, axis=1)                  
        self.s_grid = np.concatenate(([0.0], np.cumsum(segL)))
        self.sf = float(self.s_grid[-1])

        theta = np.arctan2(diffs[:, 1], diffs[:, 0])          
        self.theta_grid = np.concatenate((theta, [theta[-1]]))

        self._last_idx = 0

    @staticmethod
    def _densify(wp: np.ndarray, step: float) -> np.ndarray:
        """Linear interpolation along each segment so no gap exceeds `step`."""
        dense = [wp[0]]
        for i in range(len(wp) - 1):
            a, b = wp[i], wp[i + 1]
            seg = b - a
            L = float(np.linalg.norm(seg))
            if L < 1e-12:
                continue  # skip zero-length hops
            n = max(2, int(np.ceil(L / step)))  # include end point
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
        i = max(0, min(i, len(self.s_grid) - 2))
        s0, s1 = self.s_grid[i], self.s_grid[i + 1]
        a, b = self.path[i], self.path[i + 1]
        r = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        pos = (1.0 - r) * a + r * b
        th = self.theta_grid[i]
        return float(pos[0]), float(pos[1]), float(th)