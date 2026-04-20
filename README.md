# p73_cc

Custom controller package for the P73 walker (Bluerobin / DYROS).
This branch (`bluerobin`) targets real-robot deployment.

## 1. Setup

Copy the `p73_cc` package into your ROS 2 workspace `src` directory:

```bash
# Example layout
~/ros2_ws/src/p73_cc
```

Companion packages required in the same workspace (`~/ros2_ws/src/`):

- `p73_walker_controller` (provides `p73_controller` with the launch files)
- `p73_walker_description`
- `p73_walker_gui`
- `mujoco_ros2_sim` (only for simulation)

## 2. Switch to real-robot mode

Before building, edit `src/p73_cc/include/cc.h` and set the flag to `true`:

```cpp
// include/cc.h
bool is_on_robot_ = true;   // false = simulation, true = real robot
```

## 3. Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash      # or your ROS 2 distro
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```

## 4. Launch the controller

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch p73_controller realrobot.launch.py
```

For simulation mode (set `is_on_robot_ = false` first):

```bash
ros2 launch p73_controller simulation.launch.py
```

## 5. Send commands via keyboard teleop

In a second terminal:

```bash
python3 ~/ros2_ws/src/p73_cc/scripts/walker_teleop.py
```

Use the keyboard to send velocity/gait commands to the robot.

## Policy

The ONNX policy used by the controller is loaded from `policy/policy.onnx`.
Replace this file (or point the controller to another path) to swap policies.
