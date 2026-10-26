"""
aircraft.py -- point-mass climb/cruise model and from-scratch time integrators
AE 370 Project 1.  No external ODE solver is used.

State y = [V, h, x, m]  (airspeed m/s, altitude m, ground distance m, mass kg)
"""
import math

G0 = 9.80665

# ----------------------------------------------------------------------------
# Parameters (illustrative widebody twin-jet; NOT calibrated to a real aircraft)
# ----------------------------------------------------------------------------
P_DEFAULT = dict(
    S=550.0,            # wing area, m^2
    CD0=0.020,          # zero-lift drag coefficient
    k=0.045,            # induced drag factor
    Isp=6000.0,         # effective specific impulse, s
    m0=400000.0,        # takeoff mass, kg
    m_dry=210000.0,     # zero-fuel mass (structure + payload), kg
    CLmax=1.2,          # max usable lift coefficient (buffet-limited)
    Mmax=0.85,          # cruise/climb Mach limit
    Tmax=600e3,         # available thrust, N (held constant with altitude)
    Kspd=0.05,          # speed-hold gain, 1/s
)


# ----------------------------------------------------------------------------
# Atmosphere (NASA Glenn piecewise model, continuity-adjusted, valid 0-25 km)
# ----------------------------------------------------------------------------
def atmosphere(h, P=None):
    """Return T [C], rho [kg/m^3], a [m/s]."""
    if P is not None and "rho_const" in P:           # test hook
        return 15.04, P["rho_const"], 340.0
    # NASA Glenn model, adjusted so that T and p are continuous at 11 km
    # (the published two-layer fit has a ~0.1 % density jump there, which would
    # cap the integrator's convergence order for any climb through 11 km).
    T11 = 15.04 - 0.00649 * 11000.0
    p11 = 101.29 * ((T11 + 273.1) / 288.08) ** 5.256
    if h <= 11000.0:
        T = 15.04 - 0.00649 * h
        p = 101.29 * ((T + 273.1) / 288.08) ** 5.256
    else:
        T = T11
        p = p11 * math.exp(-0.000157 * (h - 11000.0))
    rho = p / (0.2869 * (T + 273.1))                  # p in kPa
    a = math.sqrt(1.4 * 287.05 * (T + 273.15))
    return T, rho, a


# ----------------------------------------------------------------------------
# Aerodynamics, target speed, thrust command
# ----------------------------------------------------------------------------
def lift_drag(V, h, m, gam, P):
    rho = atmosphere(h, P)[1]
    CL = 2.0 * m * G0 * math.cos(gam) / (rho * V * V * P["S"])
    D = 0.5 * rho * V * V * P["S"] * (P["CD0"] + P["k"] * CL * CL)
    return D, CL


def target_speed(h, m, P):
    """Min-drag speed at this altitude/mass, capped at the Mach limit."""
    if "V_fixed" in P:                                # test hook
        return P["V_fixed"]
    _, rho, a = atmosphere(h, P)
    V_md = math.sqrt(2.0 * m * G0 / (rho * P["S"])) * (P["k"] / P["CD0"]) ** 0.25
    return min(V_md, P["Mmax"] * a)


def thrust_command(y, gam, P):
    """Thrust that holds the flight-path angle and tracks the target speed."""
    V, h, x, m = y
    D, _ = lift_drag(V, h, m, gam, P)
    return D + m * G0 * math.sin(gam) + m * P["Kspd"] * (target_speed(h, m, P) - V)


def rhs(y, gam, P):
    """dy/dt for a flight-path angle gam (rad). gam > 0 climb, gam = 0 cruise."""
    V, h, x, m = y
    D, _ = lift_drag(V, h, m, gam, P)
    T = thrust_command(y, gam, P)
    return [(T - D) / m - G0 * math.sin(gam),
            V * math.sin(gam),
            V * math.cos(gam),
            -T / (P["Isp"] * G0)]


def feasibility(y, gam, P):
    """Return None if feasible else a string naming the violated limit."""
    V, h, x, m = y
    if V <= 1.0:
        return "stall"
    _, CL = lift_drag(V, h, m, gam, P)
    if CL > P["CLmax"]:
        return "CLmax"
    if thrust_command(y, gam, P) > P["Tmax"]:
        return "thrust"
    if m <= P["m_dry"]:
        return "fuel"
    return None


# ----------------------------------------------------------------------------
# Time integrators (from scratch)
# ----------------------------------------------------------------------------
def euler_step(y, dt, gam, P):
    f = rhs(y, gam, P)
    return [yi + dt * fi for yi, fi in zip(y, f)]


def rk4_step(y, dt, gam, P):
    k1 = rhs(y, gam, P)
    k2 = rhs([yi + 0.5 * dt * ki for yi, ki in zip(y, k1)], gam, P)
    k3 = rhs([yi + 0.5 * dt * ki for yi, ki in zip(y, k2)], gam, P)
    k4 = rhs([yi + dt * ki for yi, ki in zip(y, k3)], gam, P)
    return [yi + dt / 6.0 * (a + 2 * b + 2 * c + d)
            for yi, a, b, c, d in zip(y, k1, k2, k3, k4)]


STEPPERS = {"euler": euler_step, "rk4": rk4_step}


def _land_on_event(step, y, dt, gam, P, index, target):
    """Shortened step size tau in (0, dt] so that y[index] == target
    (bisection on the step size; each trial is one full integrator step)."""
    lo, hi = 0.0, dt
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if step(y, mid, gam, P)[index] >= target:
            hi = mid
        else:
            lo = mid
    return hi


# ----------------------------------------------------------------------------
# Flight simulation: climb until h = h_cruise, cruise until x = X
# ----------------------------------------------------------------------------
def fly(gam, h_cruise, X, P=None, dt_climb=2.0, dt_cruise=10.0,
        method="rk4", record=False):
    """Fly one trip. Returns a dict; res['ok'] False means the case is masked."""
    P = {**P_DEFAULT, **(P or {})}
    step = STEPPERS[method]
    V0 = P.get("V0", target_speed(0.0, P["m0"], P))
    y = [V0, 0.0, 0.0, P["m0"]]
    t = 0.0
    hist = [(t, *y)] if record else None

    def fail(reason):
        return dict(ok=False, reason=reason, fuel=float("nan"), t=t, y=y)

    # ---- climb ----
    reason = feasibility(y, gam, P)
    if reason:
        return fail(reason)
    while y[1] < h_cruise:
        yn = step(y, dt_climb, gam, P)
        if yn[1] >= h_cruise:
            tau = _land_on_event(step, y, dt_climb, gam, P, 1, h_cruise)
            yn = step(y, tau, gam, P)
            yn[1] = h_cruise
            dtt = tau
        else:
            dtt = dt_climb
        y, t = yn, t + dtt
        if record:
            hist.append((t, *y))
        reason = feasibility(y, gam, P)
        if reason:
            return fail(reason)
        if t > 5e4:
            return fail("timeout")
    climb = dict(t=t, x=y[2], fuel=P["m0"] - y[3], V=y[0])

    # ---- cruise ----
    while y[2] < X:
        yn = step(y, dt_cruise, 0.0, P)
        if yn[2] >= X:
            tau = _land_on_event(step, y, dt_cruise, 0.0, P, 2, X)
            yn = step(y, tau, 0.0, P)
            dtt = tau
        else:
            dtt = dt_cruise
        y, t = yn, t + dtt
        if record:
            hist.append((t, *y))
        reason = feasibility(y, 0.0, P)
        if reason:
            return fail(reason)
    return dict(ok=True, reason=None, fuel=P["m0"] - y[3], t=t, y=y,
                climb=climb, hist=hist)
