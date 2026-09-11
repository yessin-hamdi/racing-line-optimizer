# Racing Line Optimizer

A minimum-curvature trajectory optimizer that computes an optimal racing
line through a racetrack and estimates lap time, given only the track's
centerline geometry and width — built as a personal project to explore
optimization methods applied to autonomous/motorsport systems.

## What it does

Given a track's centerline coordinates and left/right width at each point
(from real F1/DTM circuit data), the optimizer finds a path that minimizes
overall curvature while staying within track boundaries (accounting for
vehicle width) — approximating a fast racing line: cutting corners toward
the inside, using the full width of the track rather than following the
centerline.

A simplified physics model then estimates a feasible speed profile along
that line (cornering-grip limit, acceleration/braking limits, top-speed
cap), from which lap time, average speed, and other metrics are computed.

## Method

- Each point on the racing line is expressed as an offset (`alpha`) from
  the centerline, along the local perpendicular direction — reducing the
  problem to one variable per track point, bounded by the available track
  width on each side (minus vehicle width, so the whole car stays on track).
- Curvature is approximated via discrete "bending energy" (squared second
  differences of position), expressed as a quadratic function of `alpha`.
- The resulting bounded quadratic optimization problem is solved using
  `scipy.optimize.minimize` (L-BFGS-B), with an analytically-derived
  gradient for fast, exact convergence.
- Speed is estimated per point from lateral grip (`v = sqrt(a_y_max * R)`),
  then refined with a forward-backward acceleration/braking sweep (run over
  several loop iterations, since the track is a closed loop) and a top-speed
  cap.
- Lap time is estimated as `T = sum(ds_i / v_i)` over the whole lap.

This approach is inspired by the method described in Heilmeier et al.,
*"Minimum Curvature Trajectory Planning and Control for an Autonomous
Race Car"* (2020), though the implementation here (offset formulation,
cost derivation, and solver setup) is my own. TUM's own reference
implementation (`TUMFTM/trajectory_planning_helpers`) additionally uses
cubic splines for geometry, iterative refinement (IQP), and a dedicated
QP solver — differences worth noting as possible future refinements here.

## Benchmark results

Optimized racing line vs. raw centerline, across 5 structurally different
real F1 tracks (simplified physics model: no aerodynamic downforce,
constant lateral grip limit):

| Track | Length (m) | Lap Time (s) | Avg Speed (km/h) | Improvement |
|---|---|---|---|---|
| Monza | 5757.91 | 91.55 | 226.41 | 7.14% |
| Nürburgring | 5056.57 | 102.71 | 177.23 | 10.24% |
| Budapest | 4315.34 | 94.52 | 164.36 | 8.52% |
| Silverstone | 5810.38 | 109.69 | 190.69 | 13.13% |
| Spa | 6948.80 | 123.90 | 201.90 | 11.29% |

**Average improvement: ~10%.** Notably, tracks with more sustained,
technical corner sequences (Silverstone, Spa) benefit more from the
optimization than tracks dominated by straights (Monza) — consistent
with the fact that minimum-curvature optimization has more room to help
wherever there's more curvature to smooth out.

## Data

Track geometry is sourced from TUM's open-source
[racetrack-database](https://github.com/TUMFTM/racetrack-database)
(F1/DTM circuit centerlines and widths).

## Requirements

- Python 3
- numpy, scipy, pandas, matplotlib

## Usage

```bash
python load_track.py [TrackName]
```

Defaults to Monza if no track is given. Computes the minimum-curvature
racing line, estimates its speed profile and lap time, compares it
against the centerline baseline, and saves plots as PNGs.

## Status

Working, validated proof of concept across multiple tracks. Known
simplifications: no aerodynamic downforce, constant tire grip assumption,
single-pass (non-iterative) curvature optimization. Planned next steps:
exploring a machine-learning surrogate model trained to approximate the
optimizer's output directly from track geometry.

## Author

Yessin Hamdi