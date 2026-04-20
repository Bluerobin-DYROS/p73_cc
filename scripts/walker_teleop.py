#!/usr/bin/env python3
"""
Walker keyboard teleop — publishes geometry_msgs/Twist to /p73/cmd_vel

Keys:
  w / s  : vx  +0.1 / -0.1  (forward/backward)
  a / d  : vy  +0.1 / -0.1  (left/right strafe)
  q / e  : wz  +0.1 / -0.1  (rotate left/right)
  space  : reset all to zero
  Ctrl+C : quit

Usage:
  ros2 run p73_cc walker_teleop.py
  # or directly:
  python3 walker_teleop.py
"""

import sys
import select
import tty
import termios

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

STEP = 0.1
MAX_V = 1.0
MAX_WZ = 0.6
PUB_HZ = 20.0

KEY_BINDINGS = {
    'w': ('vx', +STEP),
    's': ('vx', -STEP),
    'a': ('vy', +STEP),
    'd': ('vy', -STEP),
    'q': ('wz', +STEP),
    'e': ('wz', -STEP),
}

BANNER = """
=== Walker Keyboard Teleop ===
  max_v={max_v}  max_wz={max_wz}  step={step}
  w/s : forward/back  |  a/d : left/right
  q/e : rotate        |  space : stop
  Ctrl+C : quit
==============================
""".format(max_v=MAX_V, max_wz=MAX_WZ, step=STEP)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_key(settings, timeout=0.05):
    """Read a single key with timeout. Robust over SSH."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    # Save terminal settings BEFORE anything else
    settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    node = Node('walker_teleop')
    pub = node.create_publisher(Twist, '/p73/cmd_vel', 10)

    vx = 0.0
    vy = 0.0
    wz = 0.0

    print(BANNER)

    try:
        while rclpy.ok():
            key = get_key(settings, timeout=1.0 / PUB_HZ)

            if key == '\x03':  # Ctrl+C
                break
            elif key == ' ':
                vx = vy = wz = 0.0
            elif key.lower() in KEY_BINDINGS:
                axis, delta = KEY_BINDINGS[key.lower()]
                if axis == 'vx':
                    vx = clamp(vx + delta, -MAX_V, MAX_V)
                elif axis == 'vy':
                    vy = clamp(vy + delta, -MAX_V, MAX_V)
                elif axis == 'wz':
                    wz = clamp(wz + delta, -MAX_WZ, MAX_WZ)

            # Publish every loop iteration (at ~PUB_HZ)
            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.angular.z = wz
            pub.publish(msg)

            sys.stdout.write(
                f"\r  vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}   "
            )
            sys.stdout.flush()

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Restore terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        # Send zero on exit
        msg = Twist()
        pub.publish(msg)
        node.destroy_node()
        rclpy.shutdown()
        print("\nStopped.")


if __name__ == '__main__':
    main()
