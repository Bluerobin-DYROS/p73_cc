#!/usr/bin/env python3
"""
P73 Walker data analysis tool.
Compare real robot vs MuJoCo logs to diagnose sim-to-real issues.

Usage:
  # Plot single log
  python3 analyze_log.py /tmp/p73_realrobot_log.csv

  # Compare real vs sim
  python3 analyze_log.py /tmp/p73_realrobot_log.csv /tmp/p73_mujoco_log.csv

  # Quick IMU bias check
  python3 analyze_log.py /tmp/p73_realrobot_log.csv --check-bias
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_log(path):
    df = pd.read_csv(path)
    df["time"] = df["time"] - df["time"].iloc[0]  # zero-base time
    return df


def plot_imu(dfs, labels, save_dir):
    """Angular velocity and projected gravity."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    fig.suptitle("IMU: Angular Velocity (body) & Projected Gravity", fontsize=14)

    ang_cols = ["ang_vel_bx", "ang_vel_by", "ang_vel_bz"]
    grav_cols = ["proj_grav_x", "proj_grav_y", "proj_grav_z"]
    axis_names = ["X (Roll)", "Y (Pitch)", "Z (Yaw)"]

    for i, col in enumerate(ang_cols):
        ax = axes[0, i]
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[col], label=label, alpha=0.8)
        ax.set_title(f"ang_vel {axis_names[i]}")
        ax.set_ylabel("rad/s")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for i, col in enumerate(grav_cols):
        ax = axes[1, i]
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[col], label=label, alpha=0.8)
        ax.set_title(f"proj_gravity {axis_names[i]}")
        ax.set_ylabel("")
        ax.set_xlabel("time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "imu_comparison.png", dpi=150)
    print(f"  Saved: {save_dir / 'imu_comparison.png'}")


def plot_joint_pos(dfs, labels, save_dir):
    """Joint positions (relative to default)."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle("Joint Positions (relative to default)", fontsize=14)

    joint_names = [
        "L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
        "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    ]

    for i in range(12):
        ax = axes[i // 3, i % 3]
        col = f"q_rel_{i}"
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[col], label=label, alpha=0.8)
        ax.set_title(joint_names[i])
        ax.set_ylabel("rad")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "joint_pos.png", dpi=150)
    print(f"  Saved: {save_dir / 'joint_pos.png'}")


def plot_joint_vel(dfs, labels, save_dir):
    """Joint velocities."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle("Joint Velocities", fontsize=14)

    joint_names = [
        "L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
        "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    ]

    for i in range(12):
        ax = axes[i // 3, i % 3]
        col = f"qdot_{i}"
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[col], label=label, alpha=0.8)
        ax.set_title(joint_names[i])
        ax.set_ylabel("rad/s")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "joint_vel.png", dpi=150)
    print(f"  Saved: {save_dir / 'joint_vel.png'}")


def plot_torque(dfs, labels, save_dir):
    """Torque: joint space vs motor space."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle("Torque (joint space vs motor space)", fontsize=14)

    joint_names = [
        "L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
        "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    ]

    for i in range(12):
        ax = axes[i // 3, i % 3]
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[f"tau_joint_{i}"], label=f"{label} joint", alpha=0.7)
            ax.plot(df["time"], df[f"tau_motor_{i}"], label=f"{label} motor",
                    linestyle="--", alpha=0.7)
        ax.set_title(joint_names[i])
        ax.set_ylabel("Nm")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "torque.png", dpi=150)
    print(f"  Saved: {save_dir / 'torque.png'}")


def plot_actions(dfs, labels, save_dir):
    """RL actions."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle("RL Actions", fontsize=14)

    joint_names = [
        "L_HipRoll", "L_HipPitch", "L_HipYaw", "L_Knee", "L_AnklePitch", "L_AnkleRoll",
        "R_HipRoll", "R_HipPitch", "R_HipYaw", "R_Knee", "R_AnklePitch", "R_AnkleRoll",
    ]

    for i in range(12):
        ax = axes[i // 3, i % 3]
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[f"action_{i}"], label=label, alpha=0.8)
        ax.set_title(joint_names[i])
        ax.set_ylabel("action")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "actions.png", dpi=150)
    print(f"  Saved: {save_dir / 'actions.png'}")


def plot_obs_comparison(dfs, labels, save_dir):
    """Policy observation (47D newest frame) side-by-side."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Policy Observation (47D frame) — first 3 seconds", fontsize=14)

    obs_groups = [
        ("obs 0-10: ang_vel/grav/cmd/gait", range(0, 11)),
        ("obs 11-34: joint_pos_rel + joint_vel_scaled", range(11, 35)),
        ("obs 35-46: last_action", range(35, 47)),
    ]

    for ax, (title, idxs) in zip(axes, obs_groups):
        for df, label in zip(dfs, labels):
            df_short = df[df["time"] <= 3.0]
            for i in idxs:
                col = f"obs_{i}"
                ax.plot(df_short["time"], df_short[col],
                        label=f"{label} obs_{i}" if i == list(idxs)[0] else None,
                        alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(save_dir / "obs_comparison.png", dpi=150)
    print(f"  Saved: {save_dir / 'obs_comparison.png'}")


def plot_quaternion(dfs, labels, save_dir):
    """Base orientation quaternion over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Base Orientation Quaternion", fontsize=14)

    for i, (comp, name) in enumerate(
        [("quat_x", "X"), ("quat_y", "Y"), ("quat_z", "Z"), ("quat_w", "W")]
    ):
        ax = axes[i // 2, i % 2]
        for df, label in zip(dfs, labels):
            ax.plot(df["time"], df[comp], label=label, alpha=0.8)
        ax.set_title(f"quat_{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "quaternion.png", dpi=150)
    print(f"  Saved: {save_dir / 'quaternion.png'}")


def check_bias(df, label):
    """Print bias statistics for first 0.5s (robot should be standing still)."""
    df_init = df[df["time"] <= 0.5]
    if len(df_init) < 10:
        df_init = df.head(100)

    print(f"\n{'='*60}")
    print(f"  Bias check: {label} (first {df_init['time'].iloc[-1]:.2f}s, {len(df_init)} samples)")
    print(f"{'='*60}")

    print("\n  Angular velocity (body frame):")
    for col, name in [("ang_vel_bx", "Roll"), ("ang_vel_by", "Pitch"), ("ang_vel_bz", "Yaw")]:
        vals = df_init[col]
        print(f"    {name:6s}: mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"min={vals.min():+.4f}  max={vals.max():+.4f}")

    print("\n  Projected gravity (body frame):")
    for col, name in [("proj_grav_x", "X"), ("proj_grav_y", "Y"), ("proj_grav_z", "Z")]:
        vals = df_init[col]
        print(f"    {name:6s}: mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"(ideal: {0.0 if name != 'Z' else -1.0})")

    print("\n  Joint velocity (should be ~0 when standing):")
    qdot_cols = [f"qdot_{i}" for i in range(12)]
    for i, col in enumerate(qdot_cols):
        vals = df_init[col]
        if abs(vals.mean()) > 0.05:
            print(f"    joint_{i}: mean={vals.mean():+.4f}  *** SUSPICIOUS ***")

    # Check obs vs raw consistency
    print("\n  Obs[0:3] vs raw ang_vel consistency:")
    for i, raw_col in enumerate(["ang_vel_bx", "ang_vel_by", "ang_vel_bz"]):
        obs_mean = df_init[f"obs_{i}"].mean()
        raw_mean = df_init[raw_col].mean()
        diff = abs(obs_mean - raw_mean)
        flag = " *** MISMATCH ***" if diff > 0.01 else ""
        print(f"    obs_{i}={obs_mean:+.4f}  raw={raw_mean:+.4f}  diff={diff:.4f}{flag}")


def summary_stats(df, label):
    """Print quick summary of the run."""
    duration = df["time"].iloc[-1]
    hz = len(df) / duration if duration > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Summary: {label}")
    print(f"{'='*60}")
    print(f"  Duration: {duration:.2f}s  |  Samples: {len(df)}  |  Rate: {hz:.0f} Hz")

    # Check if robot fell (large quaternion deviation)
    qx_max = df["quat_x"].abs().max()
    qy_max = df["quat_y"].abs().max()
    if qx_max > 0.3 or qy_max > 0.3:
        # Find when it fell
        fall_mask = (df["quat_x"].abs() > 0.3) | (df["quat_y"].abs() > 0.3)
        fall_time = df.loc[fall_mask, "time"].iloc[0] if fall_mask.any() else None
        print(f"  ** FALL DETECTED at t={fall_time:.3f}s (quat_x_max={qx_max:.3f}, quat_y_max={qy_max:.3f})")
    else:
        print(f"  No fall detected (quat_x_max={qx_max:.3f}, quat_y_max={qy_max:.3f})")

    # Torque stats
    tau_cols = [f"tau_motor_{i}" for i in range(12)]
    tau_max = max(df[col].abs().max() for col in tau_cols)
    print(f"  Max |torque_motor|: {tau_max:.1f} Nm")

    # Action stats
    act_cols = [f"action_{i}" for i in range(12)]
    act_max = max(df[col].abs().max() for col in act_cols)
    print(f"  Max |action|: {act_max:.3f}")


def main():
    parser = argparse.ArgumentParser(description="P73 Walker log analysis")
    parser.add_argument("files", nargs="+", help="CSV log file(s)")
    parser.add_argument("--check-bias", action="store_true", help="Print IMU bias analysis")
    parser.add_argument("--save-dir", default="/tmp/p73_analysis", help="Output directory for plots")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, only print stats")
    parser.add_argument("--trange", type=float, nargs=2, metavar=("T0", "T1"),
                        help="Time range to plot (seconds)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    labels = []
    for f in args.files:
        df = load_log(f)
        if args.trange:
            df = df[(df["time"] >= args.trange[0]) & (df["time"] <= args.trange[1])].copy()
            df["time"] = df["time"] - df["time"].iloc[0]
        dfs.append(df)
        label = "REAL" if "real" in f.lower() else "SIM"
        labels.append(f"{label} ({Path(f).name})")

    # Always print stats
    for df, label in zip(dfs, labels):
        summary_stats(df, label)

    if args.check_bias:
        for df, label in zip(dfs, labels):
            check_bias(df, label)

    if not args.no_plot:
        print(f"\nGenerating plots in {save_dir} ...")
        plot_imu(dfs, labels, save_dir)
        plot_joint_pos(dfs, labels, save_dir)
        plot_joint_vel(dfs, labels, save_dir)
        plot_torque(dfs, labels, save_dir)
        plot_actions(dfs, labels, save_dir)
        plot_obs_comparison(dfs, labels, save_dir)
        plot_quaternion(dfs, labels, save_dir)
        print(f"\nAll plots saved to {save_dir}/")

    plt.show()


if __name__ == "__main__":
    main()
