# Racing Line Optimizer

A minimum-curvature trajectory optimizer that computes an optimal racing
line through a racetrack, given only the track's centerline geometry and
width — built as a personal project to explore optimization methods
applied to autonomous/motorsport systems.

## What it does

Given a track's centerline coordinates and left/right width at each point
(from real F1/DTM circuit data), the optimizer finds a path that minimizes
overall curvature while staying within track boundaries — approximating
a fast racing line: cutting corners toward the inside, using the full
width of the track rather than following the centerline.

## Method

- Each point on the racing line is expressed as an offset (`alpha`) from
  the centerline, along the local perpendicular direction — reducing the
  problem to one variable per track point, bounded by the available track
  width on each side.
- Curvature is approximated via discrete "bending energy" (squared second
  differences of position), expressed as a quadratic function of `alpha`.
- The resulting bounded quadratic optimization problem is solved using
  `scipy.optimize.minimize` (L-BFGS-B), with an analytically-derived
  gradient for fast, exact convergence.

This approach is inspired by the method described in Heilmeier et al.,
*"Minimum Curvature Trajectory Planning and Control for an Autonomous
Race Car"* (2020), though the implementation here (offset formulation,
cost derivation, and solver setup) is my own.

## Data

Track geometry is sourced from TUM's open-source
[racetrack-database](https://github.com/TUMFTM/racetrack-database)
(F1/DTM circuit centerlines and widths).

## Requirements

- Python 3
- numpy, scipy, pandas, matplotlib

## Usage

```bash
python load_track.py
```

Computes and plots the minimum-curvature racing line for Monza (the
current default track), saving the result as a PNG.

## Status

Currently a working single-track proof of concept. Planned next steps:
validating on additional tracks, and exploring a machine-learning
surrogate model trained to approximate the optimizer's output directly
from track geometry.

## Author

Yessin Hamdi