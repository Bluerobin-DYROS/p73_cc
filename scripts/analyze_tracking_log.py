#!/usr/bin/env python3
"""
Plot torque, joint position, and joint velocity from a p73 tracking log
(mujoco_tracking_*.csv / realrobot_tracking_*.csv produced by cc.cpp).

Matches the actual CSV columns:
    quat_{x,y,z,w}, ang_vel_b{x,y,z}, lin_vel_b{x,y,z},
    q_raw_{0..12}, qdot_{0..12}, obs_{0..73},
    action_{0..12}, tau_joint_{0..12}, tau_motor_{0..12}

Usage:
    python3 analyze_tracking_log.py <csv> [--save-dir DIR] [--trange T0 T1]
    # default csv = newest mujoco_tracking_*.csv in src/p73_cc/logs/
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

JOINT_NAMES_P73 = [
    "L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
    "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    "WaistYaw",
]
TORQUE_LIMITS = [352, 220, 95, 220, 95, 95,
                 352, 220, 95, 220, 95, 95, 152]


def _grid_axes(title):
    fig, axes = plt.subplots(5, 3, figsize=(20, 16), sharex=True)
    fig.suptitle(title, fontsize=14)
    for k in (13, 14):
        axes[k // 3, k % 3].axis("off")
    return fig, axes


def plot_torque(df, save_dir, csv_name):
    fig, axes = _grid_axes(f"Torque — {csv_name}")
    for i in range(13):
        ax = axes[i // 3, i % 3]
        ax.plot(df["time"], df[f"tau_joint_{i}"], label="joint", alpha=0.8, lw=0.8)
        ax.plot(df["time"], df[f"tau_motor_{i}"], label="motor",
                alpha=0.8, lw=0.8, ls="--")
        ax.axhline( TORQUE_LIMITS[i], color="r", ls=":", lw=0.7, alpha=0.5)
        ax.axhline(-TORQUE_LIMITS[i], color="r", ls=":", lw=0.7, alpha=0.5)
        ax.set_title(f"[{i}] {JOINT_NAMES_P73[i]}  (lim ±{TORQUE_LIMITS[i]} Nm)")
        ax.set_ylabel("Nm")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    plt.tight_layout()
    out = save_dir / "torque.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_joint_pos(df, save_dir, csv_name):
    fig, axes = _grid_axes(f"Joint position (raw) — {csv_name}")
    for i in range(13):
        ax = axes[i // 3, i % 3]
        ax.plot(df["time"], df[f"q_raw_{i}"], label="actual", alpha=0.8, lw=0.8)
        ax.set_title(f"[{i}] {JOINT_NAMES_P73[i]}")
        ax.set_ylabel("rad")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    plt.tight_layout()
    out = save_dir / "joint_pos.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_joint_vel(df, save_dir, csv_name):
    fig, axes = _grid_axes(f"Joint velocity — {csv_name}")
    for i in range(13):
        ax = axes[i // 3, i % 3]
        ax.plot(df["time"], df[f"qdot_{i}"], alpha=0.8, lw=0.8)
        ax.set_title(f"[{i}] {JOINT_NAMES_P73[i]}")
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    plt.tight_layout()
    out = save_dir / "joint_vel.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    return out


def summary(df):
    dur = df["time"].iloc[-1] - df["time"].iloc[0]
    rate = len(df) / dur if dur > 0 else 0
    print(f"Rows={len(df)}  Duration={dur:.2f}s  Rate={rate:.0f} Hz")
    print(f"\n{'idx':<4}{'name':<14}{'|tau| max':>11}{'% lim':>8}"
          f"{'|qdot| max':>13}{'qpos range':>16}")
    for i in range(13):
        tau_m = df[f"tau_motor_{i}"].abs().max()
        qd_m  = df[f"qdot_{i}"].abs().max()
        qmin, qmax = df[f"q_raw_{i}"].min(), df[f"q_raw_{i}"].max()
        flag = "  <SAT>" if tau_m / TORQUE_LIMITS[i] >= 0.99 else ""
        print(f"{i:<4}{JOINT_NAMES_P73[i]:<14}"
              f"{tau_m:>9.1f} Nm{tau_m/TORQUE_LIMITS[i]*100:>7.1f}%"
              f"{qd_m:>10.2f} r/s  [{qmin:>+5.2f},{qmax:>+5.2f}]{flag}")


def main():
    parser = argparse.ArgumentParser(description="P73 tracking-log plotter")
    parser.add_argument("csv", nargs="?",
                        help="CSV file (default: newest mujoco_tracking_*.csv in src/p73_cc/logs/)")
    parser.add_argument("--save-dir", default=None,
                        help="Output dir (default: same dir as the csv)")
    parser.add_argument("--trange", type=float, nargs=2, metavar=("T0", "T1"),
                        help="Time range to plot (seconds)")
    args = parser.parse_args()

    if args.csv is None:
        repo_logs = Path(__file__).resolve().parents[1] / "logs"
        candidates = sorted(repo_logs.glob("mujoco_tracking_*.csv"),
                            key=os.path.getmtime, reverse=True)
        if not candidates:
            raise SystemExit(f"No mujoco_tracking_*.csv found in {repo_logs}")
        args.csv = str(candidates[0])

    csv_path = Path(args.csv).resolve()
    save_dir = Path(args.save_dir) if args.save_dir else csv_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.trange:
        t0, t1 = args.trange
        df = df[(df["time"] >= t0) & (df["time"] <= t1)].reset_index(drop=True)

    print(f"\n=== {csv_path.name} ===")
    summary(df)

    out = []
    out.append(plot_torque(df,    save_dir, csv_path.name))
    out.append(plot_joint_pos(df, save_dir, csv_path.name))
    out.append(plot_joint_vel(df, save_dir, csv_path.name))
    print("\nSaved:")
    for p in out:
        print(f"  {p}")


if __name__ == "__main__":
    main()
