# Serial robot positioning error datasets: UR5 and Barrett WAM

Laser-tracker measurements of commanded against realised end-effector positions for
two serial manipulators with fundamentally different actuation: a 6-DOF gear-driven
Universal Robots UR5 and a 7-DOF cable-driven Barrett WAM.

I collected these for my master's thesis at Tsinghua University, supervised by
Prof. Shao Zhufeng, where both anonymous reviewers graded the thesis A. The work
later became **"A graph-based hybrid position error compensation method for serial
industrial robots"**, Yao, Jacobi, Shao and Kibireva, *Applied Soft Computing* 203
(2026) 116099.

We wanted to give the robotics community something to work with. Calibration
research is short of shared measurements: most methods are validated on one arm in
one lab, which makes results difficult to compare and harder to build on. Anyone
working on robot calibration, error compensation, or learned models of robot
behaviour is welcome to use these.

- Paper: [doi.org/10.1016/j.asoc.2026.116099](https://doi.org/10.1016/j.asoc.2026.116099)
- Code: [pauljac7700/ASTGCN-robot-pose-deviation](https://github.com/pauljac7700/ASTGCN-robot-pose-deviation)

## Why these measurements are useful

Industrial robots are repeatable but not accurate. The UR5 returns to a taught point
within 0.1 mm, but its absolute position is out by about 2.6 mm on average, because
the controller's kinematic model does not match the physical arm. Around 90% of that
gap is geometric, from manufacturing tolerance and assembly misalignment. The rest
comes from joint clearance, gear backlash, thermal drift and cable elasticity, which
all depend on the state of the robot.

Most published calibration work is tested on a single rigid gear-driven arm. The
useful thing about this pair is the contrast. The WAM is cable-driven and starts at
17.8 mm of error, an order of magnitude worse, and its errors have a different
structure. A method that works on both is doing more than fitting one machine.

With these datasets, the hybrid method in the paper reached:

| Robot | DOF | Drive | Uncompensated | After hybrid compensation | Reduction |
|---|---|---|---|---|---|
| Universal Robots UR5 | 6 | Gear | 2.566 mm | 0.155 mm | 94.0% |
| Barrett WAM | 7 | Cable | 17.766 mm | 2.918 mm | 83.6% |

The UR5 figure is close to that robot's own repeatability of 0.1 mm, which is as far
as any compensation method can go.

## Contents

| File | Poses | Purpose |
|---|---|---|
| `UR5/3D_UR5_uncalibrated_grid_cleaned.csv` | 1000 | Calibration, training and validation |
| `UR5/3D_UR5_uncalibrated_random_cleaned.csv` | 20 | Held-out test |
| `WAM/3D_WAM_uncalibrated_grid_cleaned.csv` | 216 | Calibration, training and validation |
| `WAM/3D_WAM_uncalibrated_random_cleaned.csv` | 20 | Held-out test |

Grid poses are spread systematically across the operational workspace. Random poses
are sampled independently and were never used in training, so they are the real test
of generalisation. The WAM grid is smaller than the UR5's because of practical
constraints while collecting data on that platform.

All measurements are uncalibrated. They record each robot as it comes, before any
geometric or learned correction. That is deliberate, so you can evaluate your own
calibration pipeline end to end instead of inheriting ours.

## Columns

| Column | Meaning | Units |
|---|---|---|
| `step_order` | Index of the pose in the measurement sequence | integer |
| `x_t`, `y_t`, `z_t` | Commanded target position of the end-effector | mm |
| `x_dif`, `y_dif`, `z_dif` | Measured minus commanded position, the positioning error | mm |
| `joint_1` … `joint_N` | Commanded joint angles, N = 6 for the UR5 and 7 for the WAM | degrees |

The realised position is `(x_t + x_dif, y_t + y_dif, z_t + z_dif)`. Orientation was
not measured, so these are position-only datasets.

## Quick start

```python
import pandas as pd
import numpy as np

df = pd.read_csv("UR5/3D_UR5_uncalibrated_random_cleaned.csv")

joints = df[[f"joint_{i}" for i in range(1, 7)]].to_numpy()   # inputs, degrees
error  = df[["x_dif", "y_dif", "z_dif"]].to_numpy()           # targets, mm

print(f"mean absolute position error: {np.linalg.norm(error, axis=1).mean():.3f} mm")
# 2.565 mm
```

### Sanity check

Mean absolute position error straight out of each file, with no processing:

| File | Mean absolute error |
|---|---|
| `UR5/..._grid_cleaned.csv` | 2.635 mm |
| `UR5/..._random_cleaned.csv` | 2.565 mm |
| `WAM/..._grid_cleaned.csv` | 17.114 mm |
| `WAM/..._random_cleaned.csv` | 17.623 mm |

The UR5 held-out set reproduces the paper's uncompensated baseline of 2.566 mm to
three decimal places. If your loader gives you that number, you are reading the data
correctly. Check against these values before trusting what a pipeline tells you about
a new method.

## Measurement setup

Poses were recorded with a GTS3800 laser tracker and a spherically mounted
retroreflector of 1.5 inch nominal diameter. The tracker is specified to a volumetric
accuracy of 15 µm + 6 µm/m, about two orders of magnitude finer than the errors being
measured, so tracker noise is not a limiting factor. Tool-centre frame and
base-to-tracker frame calibrations were done before collection.

The UR5 was commanded through the Robotics Toolbox for Python. The WAM was commanded
through Libbarrett, its native C++ library.

## Citation

Please cite the paper:

```bibtex
@article{yao2026graph,
  title   = {A graph-based hybrid position error compensation method for serial industrial robots},
  author  = {Yao, Ming and Jacobi, Paul and Shao, Zhufeng and Kibireva, Anna},
  journal = {Applied Soft Computing},
  volume  = {203},
  pages   = {116099},
  year    = {2026},
  doi     = {10.1016/j.asoc.2026.116099}
}
```

Ming Yao and Paul Jacobi contributed equally and are co-first authors.

The data was originally collected for the master's thesis *A Hybrid Error
Compensation Method for Enhancing the Pose Accuracy of Serial Industrial Robots*,
Paul Jacobi, Department of Mechanical Engineering, Tsinghua University, 2025.

Openly published calibration datasets, in particular Landgraf et al. (2021), set the
example for how this release was prepared.

## License

MIT, see [LICENSE](LICENSE). Use them for anything, including commercial work. A
citation is appreciated.

## Contact

Paul Jacobi, [paul.jacobi@rwth-aachen.de](mailto:paul.jacobi@rwth-aachen.de).
Questions and corrections are welcome as GitHub issues.
