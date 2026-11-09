"""Risk denominators and the relation between them (Stage 2, Section 7).

x      admitted flights per hour on a lane                       [fl/h]
tau    time one flight spends on the lane = L / V                [h]
N      mean number of aircraft simultaneously on the lane         [-]
       N = x * tau   (Little's law, stationary within the hour)
H      aircraft exposure hours generated per lane-hour            [FH/h]  (= N)
g      third-party fatalities per aircraft flight hour            [1/FH]
F      expected third-party fatalities per lane-hour              [1/h]
       F = H * g = x * tau * g
Per-flight-hour risk is F / H = g.  With loss-of-control rate lambda_f and per-crash
consequence C, g = (lambda_f + r(x)) C: the failure part does NOT grow with traffic.
Comparing F (a lane-hour total) with a per-flight-hour target theta implicitly
allows exactly theta * 1 FH of exposure per hour; the resulting 'capacity'
x* = theta / (lambda_f C tau) shrinks with lane length although no individual flight
became riskier.  That comparison is therefore a disguised societal budget.
"""
from __future__ import annotations


def occupancy(x_per_h: float, length_m: float, speed_ms: float) -> float:
    return x_per_h * (length_m / speed_ms) / 3600.0


def per_flight_hour(lambda_f: float, r: float, C: float) -> float:
    return (lambda_f + r) * C


def lane_hour_fatalities(x_per_h: float, length_m: float, speed_ms: float,
                         lambda_f: float, r: float, C: float) -> float:
    return occupancy(x_per_h, length_m, speed_ms) * per_flight_hour(lambda_f, r, C)


def implied_capacity_if_totals_compared(theta: float, lambda_f: float, C: float,
                                        length_m: float, speed_ms: float) -> float:
    """Capacity obtained by (incorrectly) comparing the lane-hour total F with a
    per-flight-hour target theta, ignoring collisions."""
    tau_h = length_m / speed_ms / 3600.0
    return theta / (lambda_f * C * tau_h)
