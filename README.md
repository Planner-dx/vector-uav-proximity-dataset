# vector-uav-proximity-dataset

**Proprioceptive Proximity Perception for a Vector-Thrust Hexarotor UAV**  
*IMU + motor signals only — no extra sensors*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-PX4%20%7C%20ROS%20Noetic-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)]()

---

## Overview

This repository contains the **flight dataset**, **signal processing pipeline**, and **PINN-based proximity estimation models** for a tilted-propeller, fully-actuated hexarotor UAV operating near surfaces (ground, wall, ceiling).

The key insight: aerodynamic proximity effects (ground effect, wall effect, ceiling effect) leave measurable signatures in **standard IMU and motor command signals** — no rangefinders, cameras, or extra hardware needed.

**Three contributions:**
1. 🗂️ **Open dataset** — multi-scenario PX4 flight logs (ULG format) with labeled proximity distances
2. 🧠 **PINN models** — Physics-Informed Neural Networks for end-to-end proximity estimation
3. 🚁 **Closed-loop demo** — autonomous landing / obstacle avoidance driven by proprioceptive perception

---

## Platform

| Parameter | Value |
|-----------|-------|
| Configuration | Tilted-rotor fully-actuated hexarotor |
| Propeller | 7×4×3 tri-blade, R = 0.0889 m |
| Motor tilt angle | ≈35° (modeled as 30°) |
| Motor layout | Irregular hexagon, diagonal 0.50 m |
| Flight controller | PX4 (Offboard mode) |
| Localization | Motion capture system |
| Motor protocol | DShot (0–2047), OGE baseline ≈ 830 |

---

## Repository Structure

```
vector-uav-proximity-dataset/
│
├── data/                          # Raw flight logs & processed datasets
│   ├── raw/                       # PX4 ULG flight logs
│   │   ├── ground_effect/         # Ascending / descending / random sorties
│   │   ├── wall_effect/           # (coming soon)
│   │   └── ceiling_effect/        # (coming soon)
│   ├── dataset.pt                 # PyTorch dataset (2510 samples × 19 features)
│   ├── dataset.csv                # Same data in CSV format
│   └── scaler.pkl                 # StandardScaler fitted on training set
│
├── models/                        # Trained model weights
│   ├── mlp_baseline.pth           # MLP baseline
│   ├── pinn_a_model.pth           # PINN-A (polynomial physics constraint)
│   └── pinn_b_model.pth           # PINN-B (potential flow constraint)
│
├── scripts/                       # Data processing & training pipeline
│   ├── process_ulg.py             # Step 1: ULG → dataset.pt / dataset.csv
│   ├── train_mlp_baseline.py      # Step 2: Train MLP baseline
│   └── train_pinn.py              # Step 3: Train PINN-A / PINN-B, compare all models
│
├── flight/                        # Flight control scripts
│   ├── sortie1_ascending.py       # Data collection: ascending z/R sequence
│   ├── sortie2_descending.py      # Data collection: descending sequence
│   ├── sortie3_random.py          # Data collection: random sequence
│   ├── scenario1_vertical_landing.py   # Demo: PINN-guided vertical landing
│   ├── scenario2_horizontal_ge.py      # Demo: horizontal flight over obstacle
│   └── spawn_cube.py              # Gazebo: spawn 1m obstacle for simulation
│
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Feature Vector (19-dim)

Each sample is a 0.5 s sliding window (step 0.1 s) over raw sensor streams:

| Index | Feature | Source |
|-------|---------|--------|
| 0–2 | acc mean (x, y, z) | `sensor_combined` |
| 3–5 | acc std (x, y, z) | `sensor_combined` |
| 6–8 | gyro mean (x, y, z) | `sensor_combined` |
| 9–11 | gyro std (x, y, z) | `sensor_combined` |
| 12–17 | 6-motor DShot mean (M1–M6) | `actuator_outputs` |
| 18 | hover thrust estimate | `hover_thrust_estimate` |

Labels: `z/R` (continuous) + `{safe, warning, danger}` (3-class)

---

## Model Performance

| Metric | MLP | PINN-A | PINN-B |
|--------|-----|--------|--------|
| MAE (overall) | 0.967 | **0.835** | 0.902 |
| RMSE | 1.376 | **1.263** | 1.354 |
| Classification Acc | 92.1% | **94.0%** | 92.1% |
| MAE (z/R ≤ 3, danger) | 0.648 | 0.606 | **0.602** |
| MAE (z/R 3–5, warning) | 0.658 | 0.712 | **0.264** |
| MAE (z/R > 5, safe) | 1.968 | **1.548** | 1.864 |

**PINN-A** uses a polynomial fit to our platform's measured thrust ratio curve.  
**PINN-B** uses the potential-flow model from Garofano-Soldado et al. (2024).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Process raw flight logs

```bash
python scripts/process_ulg.py --log data/raw/ground_effect/log_56_2026-3-17.ulg
# outputs: data/dataset.pt, data/dataset.csv, data/scaler.pkl
```

### 3. Train models

```bash
# MLP baseline
python scripts/train_mlp_baseline.py

# PINN variants + comparison
python scripts/train_pinn.py --variant pinn_a   # or pinn_b
```

### 4. Run Gazebo simulation demo

```bash
# Terminal 1: PX4 SITL
cd ~/PX4-Autopilot && make px4_sitl gazebo_typhoon_h480

# Terminal 2: MAVROS
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"

# Terminal 3: spawn obstacle (scenario 2 only)
python flight/spawn_cube.py --x 3.0 --y 0.0

# Terminal 4: run demo
python flight/scenario1_vertical_landing.py --model pinn_a --sim
python flight/scenario2_horizontal_ge.py   --model pinn_a --sim --cube-x 3.0
```

**Remove `--sim` for real hardware deployment.**

---

## Dataset Details

### Ground Effect — Sortie 1 (Ascending)

Flight structure: `Cal_pre (15 s)` → 7 hover segments (20 s each) → `Cal_post (15 s)`

| z/R | z_EKF (m) | DShot mean | T_IGE / T_OGE |
|-----|-----------|------------|---------------|
| 1.5 | 0.024 | 788.9 | 1.145 |
| 2.0 | 0.066 | 789.2 | 1.143 |
| 2.5 | 0.110 | 798.5 | 1.115 |
| 3.0 | 0.157 | 806.8 | 1.092 |
| 4.0 | 0.244 | 817.5 | 1.063 |
| 5.0 | 0.333 | 824.9 | 1.045 |
| 8.0 | 0.602 | 839.8 | 1.008 |

Battery drift: +3.9% (within acceptable range). Attitude stability: Roll std = 0.27°, Pitch std = 0.36°.

---

## Roadmap

- [x] Ground effect dataset (ascending sortie)
- [ ] Ground effect dataset (descending + random sorties)
- [ ] Wall effect dataset
- [ ] Ceiling effect dataset
- [ ] Multi-sortie cross-validation
- [ ] Real-hardware closed-loop demo video

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{luodaixun2026vectoruav,
  title  = {vector-uav-proximity-dataset: Proprioceptive Proximity Perception
             for a Vector-Thrust Hexarotor UAV},
  author = {Luo, Daixun},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/vector-uav-proximity-dataset}
}
```

---

## Related Work

- Garofano-Soldado et al. (2024) — *Assessment and Modeling of the Aerodynamic Ground Effect of a Fully-Actuated Hexarotor With Tilted Propellers*, IEEE RA-L
- McKinnon & Schoellig (2019) — *Estimating and reacting to forces and torques resulting from common aerodynamic disturbances*, RAS
- Ding et al. (2023) — *Tilted-Rotor Quadrotor*, npj Robotics

---

## License

MIT License — see [LICENSE](LICENSE) for details.
