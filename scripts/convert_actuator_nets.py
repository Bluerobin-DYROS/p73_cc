#!/usr/bin/env python3
"""Convert 12 per-joint TorchScript actuator net models to a single binary weights file.

Usage:
    python convert_actuator_nets.py [--src SRC_DIR] [--dst DST_FILE]

Defaults:
    --src  ~/isaaclab5.2/isaaclab_walker/source/isaaclab_walker/isaaclab_walker/assets/data/p73_walker/actuator_nets/
    --dst  ~/ros2_ws/src/p73_cc/actuator_nets/actuator_nets.bin

Binary format (all float32, little-endian):
    For each of 12 joints (IsaacLab _LOWER_JOINT_NAMES order):
        W0: 32*6 floats   (row-major, Linear layer 0 weight)
        b0: 32 floats     (Linear layer 0 bias)
        W1: 32*32 floats  (row-major, Linear layer 2 weight)
        b1: 32 floats     (Linear layer 2 bias)
        W2: 32*32 floats  (row-major, Linear layer 4 weight)
        b2: 32 floats     (Linear layer 4 bias)
        W3: 1*32 floats   (row-major, Linear layer 6 weight)
        b3: 1 float       (Linear layer 6 bias)

    Total per joint: 192+32+1024+32+1024+32+32+1 = 2369 floats
    Total 12 joints: 28428 floats = 113712 bytes

Joint order (matches IsaacLab _LOWER_JOINT_NAMES and MuJoCo/cc.cpp order):
    0:  L_HipRoll    -> p73_left_hip_roll.pt
    1:  L_HipPitch   -> p73_left_hip_pitch.pt
    2:  L_HipYaw     -> p73_left_hip_yaw.pt
    3:  L_Knee       -> p73_left_knee_pitch.pt
    4:  L_AnklePitch -> p73_left_ankle_pitch.pt
    5:  L_AnkleRoll  -> p73_left_ankle_roll.pt
    6:  R_HipRoll    -> p73_right_hip_roll.pt
    7:  R_HipPitch   -> p73_right_hip_pitch.pt
    8:  R_HipYaw     -> p73_right_hip_yaw.pt
    9:  R_Knee       -> p73_right_knee_pitch.pt
    10: R_AnklePitch -> p73_right_ankle_pitch.pt
    11: R_AnkleRoll  -> p73_right_ankle_roll.pt
"""

import argparse
import os
import struct
import numpy as np

# Joint model files in IsaacLab _LOWER_JOINT_NAMES order
# This order matches cc.cpp joint indexing (MuJoCo/IsaacLab order)
MODEL_FILES = [
    "p73_left_hip_roll.pt",
    "p73_left_hip_pitch.pt",
    "p73_left_hip_yaw.pt",
    "p73_left_knee_pitch.pt",
    "p73_left_ankle_pitch.pt",
    "p73_left_ankle_roll.pt",
    "p73_right_hip_roll.pt",
    "p73_right_hip_pitch.pt",
    "p73_right_hip_yaw.pt",
    "p73_right_knee_pitch.pt",
    "p73_right_ankle_pitch.pt",
    "p73_right_ankle_roll.pt",
]

# Expected state_dict keys in order: Linear(6,32), Linear(32,32), Linear(32,32), Linear(32,1)
# Indices 1, 3, 5 are Softsign activations (no parameters)
WEIGHT_KEYS = ["0.weight", "0.bias", "2.weight", "2.bias",
               "4.weight", "4.bias", "6.weight", "6.bias"]

EXPECTED_SHAPES = {
    "0.weight": (32, 6),
    "0.bias":   (32,),
    "2.weight": (32, 32),
    "2.bias":   (32,),
    "4.weight": (32, 32),
    "4.bias":   (32,),
    "6.weight": (1, 32),
    "6.bias":   (1,),
}


def main():
    default_src = os.path.expanduser(
        "~/isaaclab5.2/isaaclab_walker/source/isaaclab_walker/"
        "isaaclab_walker/assets/data/p73_walker/actuator_nets/"
    )
    default_dst = os.path.expanduser(
        "~/ros2_ws/src/p73_cc/actuator_nets/actuator_nets.bin"
    )

    parser = argparse.ArgumentParser(description="Convert TorchScript actuator nets to binary")
    parser.add_argument("--src", default=default_src, help="Source directory with .pt files")
    parser.add_argument("--dst", default=default_dst, help="Output binary file path")
    parser.add_argument("--verify", action="store_true", help="Run verification after conversion")
    args = parser.parse_args()

    import torch

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)

    total_floats = 0
    with open(args.dst, "wb") as f:
        for idx, model_file in enumerate(MODEL_FILES):
            path = os.path.join(args.src, model_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}")

            model = torch.jit.load(path, map_location="cpu").eval()
            sd = model.state_dict()

            print(f"[{idx:2d}] {model_file}")
            for key in WEIGHT_KEYS:
                tensor = sd[key]
                expected = EXPECTED_SHAPES[key]
                if tuple(tensor.shape) != expected:
                    raise ValueError(f"  {key}: expected {expected}, got {tuple(tensor.shape)}")

                arr = tensor.detach().cpu().numpy().astype(np.float32)
                # Write row-major (C-contiguous), matching PyTorch's storage
                data = arr.tobytes()
                f.write(data)
                total_floats += arr.size
                print(f"  {key}: {tuple(tensor.shape)} -> {arr.size} floats")

    file_size = os.path.getsize(args.dst)
    print(f"\nWrote {args.dst}")
    print(f"  Total: {total_floats} floats, {file_size} bytes")
    print(f"  Expected: {12 * 2369} floats, {12 * 2369 * 4} bytes")

    if total_floats != 12 * 2369:
        print("WARNING: unexpected total float count!")

    if args.verify:
        print("\n=== Verification ===")
        verify(args.src, args.dst)


def verify(src_dir, bin_path):
    """Verify binary file by comparing forward pass results."""
    import torch

    # Load binary
    data = np.fromfile(bin_path, dtype=np.float32)
    offset = 0

    def read_arr(shape):
        nonlocal offset
        size = 1
        for s in shape:
            size *= s
        arr = data[offset:offset + size].reshape(shape)
        offset += size
        return arr

    def softsign(x):
        return x / (1.0 + np.abs(x))

    for idx, model_file in enumerate(MODEL_FILES):
        # Read weights from binary
        W0 = read_arr((32, 6))
        b0 = read_arr((32,))
        W1 = read_arr((32, 32))
        b1 = read_arr((32,))
        W2 = read_arr((32, 32))
        b2 = read_arr((32,))
        W3 = read_arr((1, 32))
        b3 = read_arr((1,))

        # Forward pass with numpy
        np.random.seed(42 + idx)
        x = np.random.randn(6).astype(np.float32)

        h = W0 @ x + b0
        h = softsign(h)
        h = W1 @ h + b1
        h = softsign(h)
        h = W2 @ h + b2
        h = softsign(h)
        out_np = (W3 @ h + b3)[0]

        # Forward pass with TorchScript
        path = os.path.join(src_dir, model_file)
        model = torch.jit.load(path, map_location="cpu").eval()
        with torch.no_grad():
            out_pt = model(torch.from_numpy(x).unsqueeze(0)).item()

        err = abs(out_np - out_pt)
        status = "OK" if err < 1e-5 else "MISMATCH"
        print(f"  [{idx:2d}] {model_file}: numpy={out_np:.8f}, torch={out_pt:.8f}, err={err:.2e} [{status}]")


if __name__ == "__main__":
    main()
