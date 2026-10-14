"""One Euro Filter for low-latency signal smoothing.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 2527-2530.
"""

import math
from typing import Optional


class OneEuroFilter:
    """Adaptive 1st-order low-pass filter that dynamically balances jitter reduction

    at low speeds with responsiveness at high speeds.
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ):
        self.mincutoff = max(1e-4, float(mincutoff))
        self.beta = float(beta)
        self.dcutoff = max(1e-4, float(dcutoff))
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def reset(self, t0: float, x0: float, dx0: float = 0.0) -> None:
        """Reinitialize the filter state to a new initial position and time."""
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def alpha(self, t_e: float, cutoff: float) -> float:
        """Compute the smoothing factor alpha with safe boundary checks."""
        if t_e <= 1e-6:
            return 1.0
        cutoff = max(1e-4, cutoff)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t: float, x: float) -> float:
        """Filter a new observation *x* observed at time *t*."""
        t = float(t)
        x = float(x)
        t_e = t - self.t_prev

        # Guard against zero or negative delta-time (duplicate or out-of-order timestamps)
        if t_e <= 1e-6:
            return self.x_prev

        # Estimate derivative and smooth it
        a_d = self.alpha(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Dynamic cutoff frequency based on movement speed
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    @staticmethod
    def exponential_smoothing(a: float, x: float, x_prev: float) -> float:
        return a * x + (1.0 - a) * x_prev
