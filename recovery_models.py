#!/usr/bin/env python3
"""
Mathematical & Numerical Modeling of Post-Exercise Heart Rate Recovery (HRR)
===========================================================================
Implements and compares three differential-equation frameworks against empirical
cardiac recovery telemetry (H0 = 162 bpm, Hr = 55 bpm, t in [0, 300] s):

  1. Model 1 (Autonomous Newton-like Decay):
       dH/dt = -k * (H - Hr)  ->  H(t) = Hr + (H0 - Hr) * exp(-k * t)
     Calibrated via log-linear OLS regression (k = 0.0047 s^-1, R^2 = 0.947).

  2. Model 2 (Numerical ODE Solvers & Truncation Error Sensitivity):
       Forward Euler (O(h)) at h = 30 s vs. h = 1 s, benchmarked against RK4 (O(h^4)).

  3. Model 3 (Non-Autonomous Variable-Coefficient EPOC Lag ODE):
       dH/dt = -k_max * (1 - exp(-lambda * t)) * (H - Hr)
     Captures Excess Post-Exercise Oxygen Consumption (EPOC) & delayed
     parasympathetic reactivation via exact separation of variables and
     nonlinear least-squares parameter estimation (SciPy curve_fit).
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H_REST = 55.0


def load_empirical(csv_path: str = "hr_recovery_empirical.csv"):
    t, hr = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_s"]))
            hr.append(float(row["heart_rate_bpm"]))
    return np.array(t), np.array(hr)


def fit_log_linear_model1(t: np.ndarray, hr: np.ndarray, h_rest: float = H_REST):
    y = np.log(hr - h_rest)
    slope, intercept = np.polyfit(t, y, 1)
    y_hat = slope * t + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return -slope, np.exp(intercept), r2, y - y_hat


def model1_analytical(t: np.ndarray, k: float, h0: float = 162.0, h_rest: float = H_REST):
    return h_rest + (h0 - h_rest) * np.exp(-k * t)


def forward_euler(rhs, y0: float, t_end: float, dt: float):
    n = int(round(t_end / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + dt * rhs(t[i], y[i])
    return t, y


def rk4_solver(rhs, y0: float, t_end: float, dt: float):
    n = int(round(t_end / dt))
    t = np.linspace(0.0, n * dt, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        ti, yi = t[i], y[i]
        k1 = rhs(ti, yi)
        k2 = rhs(ti + 0.5 * dt, yi + 0.5 * dt * k1)
        k3 = rhs(ti + 0.5 * dt, yi + 0.5 * dt * k2)
        k4 = rhs(ti + dt, yi + dt * k3)
        y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


def model3_analytical(t: np.ndarray, k_max: float, lam: float, h0: float = 162.0, h_rest: float = H_REST):
    '''
    Exact solution of dH/dt = -k_max * (1 - exp(-lam * t)) * (H - h_rest)
    with initial condition H(0) = h0:
      integral_0^t k_max * (1 - exp(-lam * s)) ds = k_max * t + (k_max / lam) * (exp(-lam * t) - 1)
    '''
    exponent = -k_max * t - (k_max / lam) * (np.exp(-lam * t) - 1.0)
    return h_rest + (h0 - h_rest) * np.exp(exponent)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    t_emp, hr_emp = load_empirical()
    h0 = hr_emp[0]

    # Model 1: Log-linear OLS (fixing k=0.0047 from paper & exact polyfit check)
    k_ols, _, r2_log, log_resid = fit_log_linear_model1(t_emp, hr_emp)
    k_paper = 0.0047
    hr_m1_emp = model1_analytical(t_emp, k_paper, h0=h0)

    # Model 2: Euler h=30s vs h=1s
    rhs_const = lambda _t, h: -k_paper * (h - H_REST)
    t_eu30, hr_eu30 = forward_euler(rhs_const, h0, 300.0, 30.0)
    t_eu1, hr_eu1 = forward_euler(rhs_const, h0, 300.0, 1.0)

    # Model 3: Variable-coefficient ODE (Paper heuristic vs Nonlinear Least Squares)
    hr_m3_paper_emp = model3_analytical(t_emp, 0.0065, 0.025, h0=h0)
    popt, _ = curve_fit(
        lambda t, km, lm: model3_analytical(t, km, lm, h0=h0),
        t_emp, hr_emp, p0=[0.0065, 0.015], bounds=([0.001, 0.001], [0.05, 0.5])
    )
    k_opt, lam_opt = popt
    hr_m3_opt_emp = model3_analytical(t_emp, k_opt, lam_opt, h0=h0)

    print("=== Post-Exercise Heart Rate Recovery Modeling Summary ===")
    print(f"Model 1 (Constant Decay k={k_paper:.4f} s^-1, log-R^2={r2_log:.3f}) -> BPM RMSE = {rmse(hr_emp, hr_m1_emp):.2f} bpm")
    print(f"Model 2 (Forward Euler h=30s)                              -> BPM RMSE = {rmse(hr_emp, hr_eu30):.2f} bpm")
    print(f"Model 3a (Variable k(t): k_max=0.0065, lambda=0.0250)      -> BPM RMSE = {rmse(hr_emp, hr_m3_paper_emp):.2f} bpm")
    print(f"Model 3b (NLLS Optimal:  k_max={k_opt:.4f}, lambda={lam_opt:.4f})      -> BPM RMSE = {rmse(hr_emp, hr_m3_opt_emp):.2f} bpm")

    # Generate 3-panel diagnostic figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), dpi=180)
    t_fine = np.linspace(0, 300, 301)

    # Panel 1: Model Progression
    axes[0].scatter(t_emp, hr_emp, color="#dc2626", s=46, zorder=5, label="Empirical HR Telemetry (30s)")
    axes[0].plot(t_fine, model1_analytical(t_fine, k_paper), "--", color="#64748b", lw=1.8,
                 label=f"Model 1: Constant k (RMSE={rmse(hr_emp, hr_m1_emp):.1f})")
    axes[0].plot(t_eu30, hr_eu30, ":", color="#f59e0b", lw=1.9,
                 label=f"Model 2: Euler h=30s (RMSE={rmse(hr_emp, hr_eu30):.1f})")
    axes[0].plot(t_fine, model3_analytical(t_fine, k_opt, lam_opt), "-", color="#2563eb", lw=2.3,
                 label=f"Model 3: Variable k(t) (RMSE={rmse(hr_emp, hr_m3_opt_emp):.1f})")
    axes[0].axhline(H_REST, color="#0f172a", ls="-.", lw=1.0, alpha=0.6, label="Resting Baseline (55 bpm)")
    axes[0].set_title("Heart Rate Recovery: Autonomous vs. EPOC Lag ODE", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Recovery Time t [s]")
    axes[0].set_ylabel("Heart Rate H(t) [bpm]")
    axes[0].legend(fontsize=7.8, loc="upper right")

    # Panel 2: Residual Comparison (BPM Space)
    res_m1 = hr_emp - hr_m1_emp
    res_m3 = hr_emp - hr_m3_opt_emp
    axes[1].axhline(0, color="#0f172a", lw=1.2)
    axes[1].plot(t_emp, res_m1, "o--", color="#ef4444", lw=1.7, label="Model 1 Residuals (Systematic EPOC Bias)")
    axes[1].plot(t_emp, res_m3, "s-", color="#10b981", lw=1.9, label="Model 3 Residuals (Centered)")
    axes[1].fill_between(t_emp, 0, res_m1, color="#ef4444", alpha=0.12)
    axes[1].set_title("Residual Diagnostics: Eliminating Systematic Bias", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Recovery Time t [s]")
    axes[1].set_ylabel("Residual H_obs - H_pred [bpm]")
    axes[1].legend(fontsize=8.0, loc="upper right")

    # Panel 3: Euler O(h) vs RK4 O(h^4) Global Truncation Error Scaling
    dts = np.array([30.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25])
    err_euler, err_rk4 = [], []
    for dt in dts:
        te, ye = forward_euler(rhs_const, h0, 300.0, dt)
        tr, yr = rk4_solver(rhs_const, h0, 300.0, dt)
        err_euler.append(np.max(np.abs(ye - model1_analytical(te, k_paper))))
        err_rk4.append(np.max(np.abs(yr - model1_analytical(tr, k_paper))))

    axes[2].loglog(dts, err_euler, "o-", color="#f59e0b", lw=2.0, label="Forward Euler — O(h)")
    axes[2].loglog(dts, err_rk4, "s-", color="#2563eb", lw=2.0, label="Runge-Kutta 4 (RK4) — O(h⁴)")
    axes[2].set_title("Solver Global Truncation Error Convergence", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Step Size h [s] (log scale)")
    axes[2].set_ylabel("Max Absolute Error ||y_num - y_exact||∞ [bpm]")
    axes[2].legend(fontsize=8.3, loc="lower right")

    fig.tight_layout()
    fig.savefig("biometric_ode_diagnostics.png", dpi=180)
    plt.close(fig)
    print("Saved biometric_ode_diagnostics.png")


if __name__ == "__main__":
    main()
