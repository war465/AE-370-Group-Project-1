"""
aircraft.py -- point-mass climb/cruise model and from-scratch time integrators
AE 370 Project 1.  No external ODE solver is used.

State y = [V, h, x, m]  (airspeed m/s, altitude m, ground distance m, mass kg)
"""
import math

G0 = 9.80665

# right-hand-side evaluation counters (cost metric for the work-precision study)
_NFEV = [0]          # every call to rhs()
_EVENT_FEV = [0]     # the part of those spent locating events (reported separately)

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
    _NFEV[0] += 1
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


def abm4_step(y, dt, gam, P, fh):
    """One PECE step of the 4th-order Adams-Bashforth-Moulton pair.

    fh = [f_{n-3}, f_{n-2}, f_{n-1}, f_n] are stored derivatives at equal spacing dt.
    Predict (AB4), evaluate, correct (AM4), evaluate: 2 rhs calls per step.
    Returns the corrected state and the derivative evaluated there (reused next step).
    """
    f3, f2, f1, f0 = fh
    yp = [yi + dt / 24.0 * (55.0 * a - 59.0 * b + 37.0 * c - 9.0 * d)
          for yi, a, b, c, d in zip(y, f0, f1, f2, f3)]
    fp = rhs(yp, gam, P)
    yc = [yi + dt / 24.0 * (9.0 * p + 19.0 * a - 5.0 * b + c)
          for yi, p, a, b, c in zip(y, fp, f0, f1, f2)]
    return yc, rhs(yc, gam, P)


METHODS = ("euler", "rk4", "abm4")


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
def _march_phase(y, t, gam, idx, target, dt, P, method, hist):
    """Advance one phase until y[idx] reaches target.

    The step that would overshoot is replaced by a shortened step that lands
    exactly on the event. For 'abm4' the multistep history is built fresh at
    the start of the phase (3 RK4 start-up steps), and the landing step is RK4.
    Returns (y, t, reason) with reason None if the whole phase was feasible.
    """
    one_step = STEPPERS["rk4" if method == "abm4" else method]
    fh = [rhs(y, gam, P)] if method == "abm4" else None
    while y[idx] < target:
        fn = None
        if method == "abm4" and len(fh) == 4:
            yn, fn = abm4_step(y, dt, gam, P, fh)
        else:
            yn = one_step(y, dt, gam, P)
        landed = yn[idx] >= target
        if landed:
            n0 = _NFEV[0]
            tau = _land_on_event(one_step, y, dt, gam, P, idx, target)
            yn = one_step(y, tau, gam, P)
            if idx == 1:
                yn[1] = target
            _EVENT_FEV[0] += _NFEV[0] - n0
            dtt = tau
        else:
            dtt = dt
        y, t = yn, t + dtt
        if hist is not None:
            hist.append((t, *y))
        reason = feasibility(y, gam, P)
        if reason:
            return y, t, reason
        if landed:
            break
        if idx == 1 and t > 5e4:
            return y, t, "timeout"
        if method == "abm4":
            if fn is None:                       # start-up step: derivative at new point
                fn = rhs(y, gam, P)
            fh = (fh + [fn])[-4:]
    return y, t, None


def fly(gam, h_cruise, X, P=None, dt_climb=2.0, dt_cruise=10.0,
        method="rk4", record=False):
    """Fly one trip. Returns a dict; res['ok'] False means the case is masked.
    method in {'euler', 'rk4', 'abm4'}. res['nfev'] counts right-hand-side
    evaluations used for time marching (event location is in res['nfev_events'])."""
    P = {**P_DEFAULT, **(P or {})}
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    _NFEV[0] = 0
    _EVENT_FEV[0] = 0
    V0 = P.get("V0", target_speed(0.0, P["m0"], P))
    y = [V0, 0.0, 0.0, P["m0"]]
    t = 0.0
    hist = [(t, *y)] if record else None

    def result(ok, reason, **kw):
        return dict(ok=ok, reason=reason, t=t, y=y, hist=hist,
                    nfev=_NFEV[0] - _EVENT_FEV[0], nfev_events=_EVENT_FEV[0], **kw)

    reason = feasibility(y, gam, P)
    if reason:
        return result(False, reason, fuel=float("nan"))
    y, t, reason = _march_phase(y, t, gam, 1, h_cruise, dt_climb, P, method, hist)
    if reason:
        return result(False, reason, fuel=float("nan"))
    climb = dict(t=t, x=y[2], fuel=P["m0"] - y[3], V=y[0])
    y, t, reason = _march_phase(y, t, 0.0, 2, X, dt_cruise, P, method, hist)
    if reason:
        return result(False, reason, fuel=float("nan"))
    return result(True, None, fuel=P["m0"] - y[3], climb=climb)
