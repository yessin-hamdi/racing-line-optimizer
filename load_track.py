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


def optimize_racing_line(x, y, w_left, w_right, n_x, n_y, L):
    """Solve the minimum-curvature offset (alpha) optimization problem."""
    n_points = len(x)
    c_x = L @ x
    c_y = L @ y

    def cost_and_grad(alpha):
        Dx = c_x + L @ (alpha * n_x)
        Dy = c_y + L @ (alpha * n_y)
        cost = np.sum(Dx**2 + Dy**2)
        grad = 2 * n_x * (L @ Dx) + 2 * n_y * (L @ Dy)
        return cost, grad

    bounds = list(zip(-w_right, w_left))
    alpha_init = np.zeros(n_points)

    result = minimize(
        cost_and_grad, alpha_init,
        method="L-BFGS-B", jac=True, bounds=bounds
    )
    return result


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


def main():
    csv_path = "../racing-database/racetrack-database-master/racetrack-database-master/tracks/Monza.csv"
    track_name = "Monza"

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
                f"{track_name.lower()}_racing_line.png")


if __name__ == "__main__":
    main()