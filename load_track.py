import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.optimize import minimize


def load_track(csv_path):
    """Load a TUM racetrack-database CSV into a DataFrame."""
    track = pd.read_csv(
        csv_path,
        comment="#",
        names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]
    )
    return track


def compute_edges(x, y, w_left, w_right):
    """Compute left/right track boundary points from centerline + widths."""
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    dx = x_next - x
    dy = y_next - y

    length = np.sqrt(dx**2 + dy**2)
    dx_norm = dx / length
    dy_norm = dy / length

    left_dx = -dy_norm
    left_dy = dx_norm
    right_dx = dy_norm
    right_dy = -dx_norm

    left_x = x + left_dx * w_left
    left_y = y + left_dy * w_left
    right_x = x + right_dx * w_right
    right_y = y + right_dy * w_right

    return (left_x, left_y), (right_x, right_y), (left_dx, left_dy)


def compute_curvature(x, y):
    """Compute discrete curvature at each point along a closed-loop path."""
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    x_prev, y_prev = np.roll(x, 1), np.roll(y, 1)

    dx1 = (x_next - x_prev) / 2
    dy1 = (y_next - y_prev) / 2
    dx2 = x_next - 2 * x + x_prev
    dy2 = y_next - 2 * y + y_prev

    curvature = (dx1 * dy2 - dy1 * dx2) / (dx1**2 + dy1**2)**1.5
    return curvature


def build_second_diff_matrix(n_points):
    """Build the sparse closed-loop second-difference matrix L."""
    L = diags(
        [1, -2, 1], [-1, 0, 1],
        shape=(n_points, n_points),
        dtype=float
    ).tolil()
    L[0, -1] = 1
    L[-1, 0] = 1
    return L.tocsr()

def optimize_racing_line(x, y, w_left, w_right, n_x, n_y, L, w_veh=2.0):
    n_points = len(x)
    c_x = L @ x
    c_y = L @ y

    # Shrink usable width by half the vehicle width on each side
    w_left_adj = w_left - w_veh / 2
    w_right_adj = w_right - w_veh / 2
    if np.any(w_left_adj < 0) or np.any(w_right_adj < 0):
        n_bad_left = np.sum(w_left_adj < 0)
        n_bad_right = np.sum(w_right_adj < 0)
        raise ValueError(
            f"Vehicle width ({w_veh}m) doesn't fit at some track points! "
            f"{n_bad_left} points too narrow on the left, "
            f"{n_bad_right} points too narrow on the right. "
            f"Reduce w_veh or check the track data."
        )

    def cost_and_grad(alpha):
        Dx = c_x + L @ (alpha * n_x)
        Dy = c_y + L @ (alpha * n_y)
        cost = np.sum(Dx**2 + Dy**2)
        grad = 2 * n_x * (L @ Dx) + 2 * n_y * (L @ Dy)
        return cost, grad

    bounds = list(zip(-w_right_adj, w_left_adj))
    alpha_init = np.zeros(n_points)

    result = minimize(
        cost_and_grad, alpha_init,
        method="L-BFGS-B", jac=True, bounds=bounds
    )
    return result
def compute_speed_profile(x, y, a_y_max=15.0):
    """
    Compute the maximum cornering speed at each point along a path,
    based purely on lateral grip limits (v_max = sqrt(a_y_max * R)).
    Does NOT account for engine/braking limits - straights will show
    unrealistically high values, since this only captures the
    cornering constraint.
    """
    curvature = compute_curvature(x, y)

    # Avoid division by zero on perfectly straight sections
    # (curvature exactly 0 would make R infinite).
    # np.abs() since curvature sign indicates turn direction (left/right),
    # but radius itself is always a positive distance.
    curvature_safe = np.maximum(np.abs(curvature), 1e-6)

    R = 1 / curvature_safe
    v_max = np.sqrt(a_y_max * R)

    return v_max, R
def apply_acceleration_limits(v_max, x, y, a_max=10.0, a_brake=25.0, v_top_speed=100.0, n_laps=3):
    n = len(v_max)
    v_profile = np.minimum(v_max, v_top_speed).copy()

    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    ds = np.sqrt((x_next - x)**2 + (y_next - y)**2)

    # Repeat the forward-backward sweep several times so the seam
    # between the last point and point 0 (the closed loop) converges
    # to a consistent value, instead of being treated as a hard start.
    for _ in range(n_laps):
        for i in range(n):
            i_prev = (i - 1) % n  # wraps around: point 0's previous is point n-1
            v_reachable = np.sqrt(v_profile[i_prev]**2 + 2 * a_max * ds[i_prev])
            v_profile[i] = min(v_profile[i], v_reachable)

        for i in range(n - 1, -1, -1):
            i_next = (i + 1) % n  # wraps around: point n-1's next is point 0
            v_reachable = np.sqrt(v_profile[i_next]**2 + 2 * a_brake * ds[i])
            v_profile[i] = min(v_profile[i], v_reachable)

    return v_profile

def plot_result(x, y, left, right, racing_x, racing_y, track_name, save_path):
    """Plot centerline, edges, and optimized racing line."""
    plt.figure(figsize=(10, 8))
    plt.plot(x, y, linewidth=1, color="blue", alpha=0.5, label="Centerline")
    plt.plot(*left, linewidth=1, color="green", label="Left edge")
    plt.plot(*right, linewidth=1, color="red", label="Right edge")
    plt.plot(racing_x, racing_y, linewidth=1.5, color="black", label="Optimized racing line")
    plt.axis("equal")
    plt.title(f"{track_name} — Minimum-Curvature Racing Line")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
def compute_metrics(x, y, v_profile):
    """
    Compute summary metrics for a path + its speed profile:
    total length, lap time, average speed, min/max speed.
    """
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    ds = np.sqrt((x_next - x)**2 + (y_next - y)**2)

    length = np.sum(ds)
    lap_time = np.sum(ds / v_profile)
    avg_speed = length / lap_time

    return {
        "length_m": length,
        "lap_time_s": lap_time,
        "avg_speed_kmh": avg_speed * 3.6,
        "max_speed_kmh": v_profile.max() * 3.6,
        "min_speed_kmh": v_profile.min() * 3.6,
    }


import sys


def main():
    track_name = sys.argv[1] if len(sys.argv) > 1 else "Monza"
    csv_path = f"../racing-database/racetrack-database-master/racetrack-database-master/tracks/{track_name}.csv"

    track = load_track(csv_path)
    x = track["x_m"].values
    y = track["y_m"].values
    w_right = track["w_tr_right_m"].values
    w_left = track["w_tr_left_m"].values

    left, right, (n_x, n_y) = compute_edges(x, y, w_left, w_right)

    curvature = compute_curvature(x, y)
    print(f"Max curvature: {curvature.max():.5f}")
    print(f"Min curvature: {curvature.min():.5f}")

    L = build_second_diff_matrix(len(x))

    result = optimize_racing_line(x, y, w_left, w_right, n_x, n_y, L)
    print("Optimization success:", result.success)
    print("Message:", result.message)
    print("Iterations:", result.nit)
    print("Final cost:", result.fun)

    alpha_opt = result.x
    racing_x = x + alpha_opt * n_x
    racing_y = y + alpha_opt * n_y

    plot_result(x, y, left, right, racing_x, racing_y, track_name,
                f"results/{track_name.lower()}_racing_line.png")

    v_max, R = compute_speed_profile(racing_x, racing_y)
    v_profile = apply_acceleration_limits(v_max, racing_x, racing_y)
    metrics = compute_metrics(racing_x, racing_y, v_profile)
    print("\n--- Metrics (optimized racing line) ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
        # --- Compare against the centerline (unoptimized baseline) ---
    v_max_centerline, _ = compute_speed_profile(x, y)
    v_profile_centerline = apply_acceleration_limits(v_max_centerline, x, y)
    metrics_centerline = compute_metrics(x, y, v_profile_centerline)

    print("\n--- Metrics (centerline, unoptimized) ---")
    for key, value in metrics_centerline.items():
        print(f"{key}: {value:.2f}")

    improvement = (metrics_centerline["lap_time_s"] - metrics["lap_time_s"]) / metrics_centerline["lap_time_s"] * 100
    print(f"\nLap time improvement from optimization: {improvement:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.plot(v_max * 3.6, label="Cornering limit only", alpha=0.5)
    plt.plot(v_profile * 3.6, label="With acceleration/braking limits", linewidth=1.5)
    plt.title(f"{track_name} — Speed Profile")
    plt.xlabel("Point index (along track)")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.savefig(f"results/{track_name.lower()}_racing_line.png")
    plt.show()

if __name__ == "__main__":
    main()