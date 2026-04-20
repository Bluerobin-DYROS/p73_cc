#!/usr/bin/env python3
"""Auto-plot all signals from P73 robot CSV logs.

Usage:
  python3 plot_log.py                                         # latest CSV
  python3 plot_log.py --csv <path.csv>                        # specific CSV
  python3 plot_log.py --compare mujoco_xxx.csv realrobot_xxx.csv  # sim vs real overlay
  python3 plot_log.py --watch                                 # auto-plot new CSVs
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_BASE_DIR = SCRIPT_DIR / "plot"
INDEXED_COL_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")

JOINT_NAMES = [
    "L_HipRoll", "L_HipPitch", "L_HipYaw",
    "L_Knee", "L_AnklePitch", "L_AnkleRoll",
    "R_HipRoll", "R_HipPitch", "R_HipYaw",
    "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    "WaistYaw",
]

# Colors for compare mode
SIM_COLOR = "#1f77b4"   # blue
REAL_COLOR = "#d62728"   # red
SIM_ALPHA = 0.8
REAL_ALPHA = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-plot P73 robot log CSV")
    parser.add_argument("--csv", type=Path, default=None, help="CSV file path (default: latest)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("CSV1", "CSV2"),
                        help="Compare two CSVs (sim vs real) overlaid on same plots")
    parser.add_argument("--align", choices=["time", "start"], default="start",
                        help="Time alignment for --compare: 'start' aligns t=0 (default), "
                             "'time' uses raw timestamps")
    parser.add_argument("--show", action="store_true", help="Open plot windows")
    parser.add_argument("--watch", action="store_true",
                        help="Watch logs/ for new CSV files and plot automatically")
    parser.add_argument("--max-subplots", type=int, default=13)
    return parser.parse_args()


def find_latest_csv(log_dir: Path, prefix: Optional[str] = None) -> Path:
    if prefix:
        patterns = [f"{prefix}_*.csv"]
    else:
        patterns = ["realrobot_*.csv", "mujoco_*.csv"]
    candidates = []
    for pat in patterns:
        candidates.extend(log_dir.glob(pat))
    candidates.sort(key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No CSV logs found in {log_dir}")
    return candidates[-1]


def resolve_csv(path: Path) -> Path:
    if path.is_absolute():
        return path
    # Try as-is first, then relative to SCRIPT_DIR
    if path.exists():
        return path.resolve()
    candidate = SCRIPT_DIR / path
    if candidate.exists():
        return candidate.resolve()
    # Try adding .csv extension
    if not path.suffix:
        candidate = SCRIPT_DIR / (path.name + ".csv")
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def load_csv(csv_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            if len(row) < ncols:
                row = row + ["nan"] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            rows.append([safe_float(v) for v in row])

    arr = np.array(rows, dtype=float)
    data = {name: arr[:, i] for i, name in enumerate(header)}
    if "time" in data and len(data["time"]) > 0:
        data["time"] = data["time"] - data["time"][0]
    return header, data


def detect_label(csv_path: Path) -> str:
    name = csv_path.stem.lower()
    if "mujoco" in name or "sim" in name:
        return "sim"
    return "real"


def joint_label(prefix: str, idx: int) -> str:
    if idx < len(JOINT_NAMES):
        return f"{prefix}[{idx}] {JOINT_NAMES[idx]}"
    return f"{prefix}[{idx}]"


# ═══════════════════════════════════════════════════════════════════
# Single-CSV plots (unchanged)
# ═══════════════════════════════════════════════════════════════════

def plot_imu(t, d, out):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("IMU / Base State", fontsize=14)
    for name in ["quat_x", "quat_y", "quat_z", "quat_w"]:
        if name in d:
            axes[0, 0].plot(t, d[name], lw=0.8, label=name)
    axes[0, 0].set_title("Quaternion (xyzw)"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)
    for name in ["ang_vel_bx", "ang_vel_by", "ang_vel_bz"]:
        if name in d:
            axes[0, 1].plot(t, d[name], lw=0.8, label=name.split("_")[-1])
    axes[0, 1].set_title("Angular Velocity (body)"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)
    for name in ["proj_grav_x", "proj_grav_y", "proj_grav_z"]:
        if name in d:
            axes[1, 0].plot(t, d[name], lw=0.8, label=name.split("_")[-1])
    axes[1, 0].set_title("Projected Gravity (body)"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)
    for name in ["lin_vel_wx", "lin_vel_wy", "lin_vel_wz"]:
        if name in d:
            axes[1, 1].plot(t, d[name], lw=0.8, label=name.split("_")[-1])
    axes[1, 1].set_title("Linear Velocity (world)"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)
    for name in ["cmd_vx", "cmd_vy", "cmd_vyaw"]:
        if name in d:
            axes[2, 0].plot(t, d[name], lw=0.8, label=name)
    axes[2, 0].set_title("Velocity Command"); axes[2, 0].legend(fontsize=8); axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("time [s]")
    for name in ["gait_sin", "gait_cos"]:
        if name in d:
            axes[2, 1].plot(t, d[name], lw=0.8, label=name)
    axes[2, 1].set_title("Gait Phase"); axes[2, 1].legend(fontsize=8); axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlabel("time [s]")
    if "value" in d:
        ax_val = axes[2, 1].twinx()
        ax_val.plot(t, d["value"], lw=0.8, color="red", alpha=0.5, label="value")
        ax_val.set_ylabel("value", color="red"); ax_val.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); path = out / "01_imu_base_state.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_joint_group(t, d, prefix, title, filename, out, n_joints=13):
    cols = [f"{prefix}_{i}" for i in range(n_joints) if f"{prefix}_{i}" in d]
    if not cols:
        return None
    n = len(cols); ncols = 3; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1); fig.suptitle(title, fontsize=14)
    for i, col in enumerate(cols):
        axes[i].plot(t, d[col], lw=0.8); axes[i].set_title(joint_label(prefix, i))
        axes[i].grid(True, alpha=0.3); axes[i].set_xlabel("time [s]")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout(); path = out / filename; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_joint_pos_vs_action(t, d, out):
    n = 12; ncols = 3; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1); fig.suptitle("Joint Pos Relative vs RL Action", fontsize=14)
    for i in range(n):
        ax = axes[i]
        if f"q_rel_{i}" in d:
            ax.plot(t, d[f"q_rel_{i}"], lw=0.8, label="q_rel", color="tab:blue")
        if f"action_{i}" in d:
            ax2 = ax.twinx()
            ax2.plot(t, d[f"action_{i}"], lw=0.8, label="action", color="tab:orange", alpha=0.7)
            ax2.set_ylabel("action", fontsize=8, color="tab:orange"); ax2.legend(fontsize=7, loc="upper right")
        ax.set_title(joint_label("joint", i)); ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]"); ax.legend(fontsize=7, loc="upper left")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout(); path = out / "05_joint_pos_vs_action.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_torque_comparison(t, d, out):
    n = 13; ncols = 3; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1); fig.suptitle("Torque: Joint (blue) vs Motor/4-bar (orange)", fontsize=14)
    for i in range(n):
        ax = axes[i]
        if f"tau_joint_{i}" in d:
            ax.plot(t, d[f"tau_joint_{i}"], lw=0.8, label="joint", color="tab:blue")
        if f"tau_motor_{i}" in d:
            ax.plot(t, d[f"tau_motor_{i}"], lw=0.8, label="motor", color="tab:orange", alpha=0.7)
        ax.set_title(joint_label("torque", i)); ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]"); ax.legend(fontsize=7)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout(); path = out / "06_torque_joint_vs_motor.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_motor_torque(t, d, out):
    cols = [f"tau_motor_{i}" for i in range(13) if f"tau_motor_{i}" in d]
    if not cols:
        return None
    n = 13; ncols = 3; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1); fig.suptitle("Motor Torque (4-bar)", fontsize=14)
    for i in range(n):
        ax = axes[i]
        col = f"tau_motor_{i}"
        if col in d:
            ax.plot(t, d[col], lw=0.8, color="tab:orange")
        ax.set_title(joint_label("motor_torque", i)); ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout(); path = out / "06b_motor_torque.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot_obs_frame(t, d, out):
    obs_cols = [f"obs_{i}" for i in range(47) if f"obs_{i}" in d]
    if not obs_cols:
        return None
    groups = [
        ("ang_vel (obs 0-2)", [0, 1, 2]),
        ("proj_grav (obs 3-5)", [3, 4, 5]),
        ("cmd_vel (obs 6-8)", [6, 7, 8]),
        ("gait (obs 9-10)", [9, 10]),
        ("joint_pos_rel (obs 11-22)", list(range(11, 23))),
        ("joint_vel_scaled (obs 23-34)", list(range(23, 35))),
        ("last_action (obs 35-46)", list(range(35, 47))),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(18, 3.5 * len(groups)), sharex=True)
    fig.suptitle("Policy Observation Frame (47D)", fontsize=14)
    for ax, (label, indices) in zip(axes, groups):
        for idx in indices:
            col = f"obs_{idx}"
            if col in d:
                ax.plot(t, d[col], lw=0.7, label=f"obs_{idx}", alpha=0.8)
        ax.set_title(label); ax.grid(True, alpha=0.3)
        if len(indices) <= 6:
            ax.legend(fontsize=7, ncol=len(indices))
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(); path = out / "07_obs_frame_47d.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# Compare mode: sim vs real overlaid
# ═══════════════════════════════════════════════════════════════════

def _dual_legend(ax):
    """Add a small sim/real legend to an axis."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=SIM_COLOR, lw=1.5, label="sim"),
        Line2D([0], [0], color=REAL_COLOR, lw=1.5, label="real"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")


def compare_imu(ts, ds, tr, dr, out):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("IMU / Base State  —  sim (blue) vs real (red)", fontsize=14)

    # Quaternion
    for name in ["quat_x", "quat_y", "quat_z", "quat_w"]:
        if name in ds:
            axes[0, 0].plot(ts, ds[name], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if name in dr:
            axes[0, 0].plot(tr, dr[name], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
    axes[0, 0].set_title("Quaternion (xyzw)"); _dual_legend(axes[0, 0]); axes[0, 0].grid(True, alpha=0.3)

    # Angular velocity
    for name in ["ang_vel_bx", "ang_vel_by", "ang_vel_bz"]:
        if name in ds:
            axes[0, 1].plot(ts, ds[name], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if name in dr:
            axes[0, 1].plot(tr, dr[name], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
    axes[0, 1].set_title("Angular Velocity (body)"); _dual_legend(axes[0, 1]); axes[0, 1].grid(True, alpha=0.3)

    # Projected gravity
    for name in ["proj_grav_x", "proj_grav_y", "proj_grav_z"]:
        if name in ds:
            axes[1, 0].plot(ts, ds[name], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if name in dr:
            axes[1, 0].plot(tr, dr[name], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
    axes[1, 0].set_title("Projected Gravity (body)"); _dual_legend(axes[1, 0]); axes[1, 0].grid(True, alpha=0.3)

    # Linear velocity
    for name in ["lin_vel_wx", "lin_vel_wy", "lin_vel_wz"]:
        if name in ds:
            axes[1, 1].plot(ts, ds[name], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if name in dr:
            axes[1, 1].plot(tr, dr[name], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
    axes[1, 1].set_title("Linear Velocity (world)"); _dual_legend(axes[1, 1]); axes[1, 1].grid(True, alpha=0.3)

    # Command velocity
    for name in ["cmd_vx", "cmd_vy", "cmd_vyaw"]:
        if name in ds:
            axes[2, 0].plot(ts, ds[name], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if name in dr:
            axes[2, 0].plot(tr, dr[name], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
    axes[2, 0].set_title("Velocity Command"); _dual_legend(axes[2, 0]); axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("time [s]")

    # Gait phase + value
    for name in ["gait_sin", "gait_cos"]:
        if name in ds:
            axes[2, 1].plot(ts, ds[name], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if name in dr:
            axes[2, 1].plot(tr, dr[name], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
    axes[2, 1].set_title("Gait Phase"); _dual_legend(axes[2, 1]); axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlabel("time [s]")

    fig.tight_layout(); path = out / "01_compare_imu_base.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def compare_joint_group(ts, ds, tr, dr, prefix, title, filename, out, n_joints=13):
    cols = [f"{prefix}_{i}" for i in range(n_joints)
            if f"{prefix}_{i}" in ds or f"{prefix}_{i}" in dr]
    if not cols:
        return None
    n = len(cols); ncols = 3; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"{title}  —  sim (blue) vs real (red)", fontsize=14)
    for i, col in enumerate(cols):
        ax = axes[i]
        if col in ds:
            ax.plot(ts, ds[col], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if col in dr:
            ax.plot(tr, dr[col], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
        ax.set_title(joint_label(prefix, i)); ax.grid(True, alpha=0.3); ax.set_xlabel("time [s]")
        _dual_legend(ax)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout(); path = out / filename; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def compare_obs_frame(ts, ds, tr, dr, out):
    groups = [
        ("ang_vel (obs 0-2)", [0, 1, 2]),
        ("proj_grav (obs 3-5)", [3, 4, 5]),
        ("cmd_vel (obs 6-8)", [6, 7, 8]),
        ("gait (obs 9-10)", [9, 10]),
        ("joint_pos_rel (obs 11-22)", list(range(11, 23))),
        ("joint_vel_scaled (obs 23-34)", list(range(23, 35))),
        ("last_action (obs 35-46)", list(range(35, 47))),
    ]
    has_any = any(f"obs_{i}" in ds or f"obs_{i}" in dr for i in range(47))
    if not has_any:
        return None

    fig, axes = plt.subplots(len(groups), 1, figsize=(18, 3.5 * len(groups)), sharex=True)
    fig.suptitle("Policy Obs Frame (47D)  —  sim (blue) vs real (red)", fontsize=14)
    for ax, (label, indices) in zip(axes, groups):
        for idx in indices:
            col = f"obs_{idx}"
            if col in ds:
                ax.plot(ts, ds[col], lw=0.6, color=SIM_COLOR, alpha=0.6)
            if col in dr:
                ax.plot(tr, dr[col], lw=0.6, color=REAL_COLOR, alpha=0.6)
        ax.set_title(label); ax.grid(True, alpha=0.3); _dual_legend(ax)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(); path = out / "07_compare_obs_frame.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def compare_obs_per_joint(ts, ds, tr, dr, out):
    """Per-joint overlay: each subplot shows sim vs real for one joint's obs."""
    # joint_pos_rel (obs 11-22) and joint_vel_scaled (obs 23-34)
    for obs_start, n, tag, title in [
        (11, 12, "joint_pos_rel", "Obs: Joint Pos Rel (obs 11-22)"),
        (23, 12, "joint_vel_scaled", "Obs: Joint Vel Scaled (obs 23-34)"),
        (35, 12, "last_action", "Obs: Last Action (obs 35-46)"),
    ]:
        ncols = 3; nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)
        fig.suptitle(f"{title}  —  sim (blue) vs real (red)", fontsize=14)
        for i in range(n):
            ax = axes[i]
            col = f"obs_{obs_start + i}"
            if col in ds:
                ax.plot(ts, ds[col], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
            if col in dr:
                ax.plot(tr, dr[col], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
            ax.set_title(joint_label(tag, i)); ax.grid(True, alpha=0.3)
            ax.set_xlabel("time [s]"); _dual_legend(ax)
        for i in range(n, len(axes)):
            axes[i].axis("off")
        fname = f"08_compare_{tag}.png"
        fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)
    return out


def compare_torque(ts, ds, tr, dr, out):
    n = 13; ncols = 3; nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Torque (joint space)  —  sim (blue) vs real (red)", fontsize=14)
    for i in range(n):
        ax = axes[i]
        col = f"tau_joint_{i}"
        if col in ds:
            ax.plot(ts, ds[col], lw=0.7, color=SIM_COLOR, alpha=SIM_ALPHA)
        if col in dr:
            ax.plot(tr, dr[col], lw=0.7, color=REAL_COLOR, alpha=REAL_ALPHA)
        ax.set_title(joint_label("torque", i)); ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]"); _dual_legend(ax)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout(); path = out / "06_compare_torque.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def compare_rmse_summary(ts, ds, tr, dr, out):
    """Bar chart of per-signal RMSE between sim and real (interpolated to common time)."""
    t_end = min(ts[-1], tr[-1])
    t_common = np.linspace(0, t_end, min(len(ts), len(tr), 5000))

    signal_groups = {
        "ang_vel": ["ang_vel_bx", "ang_vel_by", "ang_vel_bz"],
        "proj_grav": ["proj_grav_x", "proj_grav_y", "proj_grav_z"],
        "q_rel": [f"q_rel_{i}" for i in range(12)],
        "qdot": [f"qdot_{i}" for i in range(13)],
        "action": [f"action_{i}" for i in range(12)],
        "tau_joint": [f"tau_joint_{i}" for i in range(13)],
    }

    group_rmse = {}
    for group, cols in signal_groups.items():
        rmses = []
        for col in cols:
            if col in ds and col in dr:
                s_interp = np.interp(t_common, ts, ds[col])
                r_interp = np.interp(t_common, tr, dr[col])
                rmse = np.sqrt(np.nanmean((s_interp - r_interp) ** 2))
                rmses.append(rmse)
        if rmses:
            group_rmse[group] = np.mean(rmses)

    if not group_rmse:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(group_rmse.keys())
    vals = [group_rmse[n] for n in names]
    bars = ax.barh(names, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"])
    ax.set_xlabel("RMSE (sim vs real)")
    ax.set_title("Sim-to-Real Gap Summary (mean RMSE per signal group)")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout(); path = out / "00_compare_rmse_summary.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def run_compare(path1: Path, path2: Path, show: bool = False):
    """Compare two CSVs with sim vs real overlay."""
    path1, path2 = resolve_csv(path1), resolve_csv(path2)
    for p in [path1, path2]:
        if not p.exists():
            print(f"[error] CSV not found: {p}", file=sys.stderr); sys.exit(1)

    label1, label2 = detect_label(path1), detect_label(path2)
    # Ensure sim is first, real is second
    if label1 == "real" and label2 == "sim":
        path1, path2 = path2, path1
        label1, label2 = "sim", "real"

    print(f"[compare] sim: {path1.name}")
    print(f"[compare] real: {path2.name}")

    _, ds = load_csv(path1)
    _, dr = load_csv(path2)
    ts, tr = ds["time"], dr["time"]

    out_dir = PLOT_BASE_DIR / f"compare_{path1.stem}_vs_{path2.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []

    # RMSE summary
    p = compare_rmse_summary(ts, ds, tr, dr, out_dir)
    if p: saved.append(p)

    # IMU / base
    p = compare_imu(ts, ds, tr, dr, out_dir)
    if p: saved.append(p)

    # Joint pos raw
    p = compare_joint_group(ts, ds, tr, dr, "q_raw", "Joint Position (raw)",
                            "02_compare_joint_pos_raw.png", out_dir)
    if p: saved.append(p)

    # Joint pos relative
    p = compare_joint_group(ts, ds, tr, dr, "q_rel", "Joint Position (relative)",
                            "03_compare_joint_pos_rel.png", out_dir, n_joints=12)
    if p: saved.append(p)

    # Joint velocity
    p = compare_joint_group(ts, ds, tr, dr, "qdot", "Joint Velocity",
                            "04_compare_joint_vel.png", out_dir)
    if p: saved.append(p)

    # Actions
    p = compare_joint_group(ts, ds, tr, dr, "action", "RL Actions",
                            "05_compare_actions.png", out_dir, n_joints=12)
    if p: saved.append(p)

    # Torque
    p = compare_torque(ts, ds, tr, dr, out_dir)
    if p: saved.append(p)

    # Obs frame overview
    p = compare_obs_frame(ts, ds, tr, dr, out_dir)
    if p: saved.append(p)

    # Per-joint obs
    compare_obs_per_joint(ts, ds, tr, dr, out_dir)
    for f in sorted(out_dir.glob("08_compare_*.png")):
        saved.append(f)

    print(f"\n[compare] Saved {len(saved)} plots to: {out_dir}/")
    for p in saved:
        print(f"  {p.name}")

    if show:
        matplotlib.use("TkAgg"); plt.show()

    return out_dir


# ═══════════════════════════════════════════════════════════════════
# Single-CSV pipeline
# ═══════════════════════════════════════════════════════════════════

def plot_csv(csv_path: Path, show: bool = False) -> Path:
    print(f"[plot] Loading: {csv_path.name}")
    header, data = load_csv(csv_path)
    if "time" not in data:
        raise RuntimeError("CSV must contain a 'time' column.")
    t = data["time"]
    out_dir = PLOT_BASE_DIR / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fn in [
        lambda: plot_imu(t, data, out_dir),
        lambda: plot_joint_group(t, data, "q_raw", "Joint Position (raw)", "02_joint_pos_raw.png", out_dir),
        lambda: plot_joint_group(t, data, "qdot", "Joint Velocity", "03_joint_vel.png", out_dir),
        lambda: plot_joint_group(t, data, "action", "RL Actions", "04_actions.png", out_dir, n_joints=12),
        lambda: plot_joint_pos_vs_action(t, data, out_dir),
        lambda: plot_torque_comparison(t, data, out_dir),
        lambda: plot_motor_torque(t, data, out_dir),
        lambda: plot_obs_frame(t, data, out_dir),
    ]:
        p = fn()
        if p: saved.append(p)
    print(f"[plot] Saved {len(saved)} plots to: {out_dir}/")
    for p in saved:
        print(f"  {p.name}")
    if show:
        matplotlib.use("TkAgg"); plt.show()
    return out_dir


def watch_loop():
    print(f"[watch] Monitoring {SCRIPT_DIR} for new CSV files... (Ctrl+C to stop)")
    seen = {p.name for p in SCRIPT_DIR.glob("*.csv")}
    print(f"[watch] {len(seen)} existing CSV files skipped.")
    while True:
        for p in sorted(SCRIPT_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime):
            if p.name not in seen:
                prev_size = -1
                for _ in range(10):
                    cur_size = p.stat().st_size
                    if cur_size == prev_size and cur_size > 0:
                        break
                    prev_size = cur_size
                    time_mod.sleep(1.0)
                seen.add(p.name)
                try:
                    plot_csv(p)
                except Exception as e:
                    print(f"[watch] Error plotting {p.name}: {e}")
        time_mod.sleep(2.0)


def main():
    args = parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1], show=args.show)
        return

    if args.watch:
        watch_loop()
        return

    if args.csv is not None:
        csv_path = resolve_csv(args.csv)
    else:
        csv_path = find_latest_csv(SCRIPT_DIR)

    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    plot_csv(csv_path, show=args.show)


if __name__ == "__main__":
    main()
