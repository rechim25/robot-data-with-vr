import numpy as np
import ctypes
import xr
import carb
import time
import cv2
import time

from xr_wrapper import XrWrapper

from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        # "width": 2800,
        # "height": 2400,
        # "window_width": 2900,
        # "window_height": 2500,
        "headless": True
    }
)  # we can also run as headless.
from omni.isaac.core import World
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

CAM_RESOLUTION = (1832, 1920)  # Native Quest 2 resolution (ratio 1.9)

CAM_FREQ = 50

ORIGIN_POS = np.array([-1.3, -0.8, 0.5])

IPD_OFFSET = np.array([0, 0.3, 0])

ORIGIN_ROT = rot_utils.euler_angles_to_quats(np.array([180, 0, 25]), degrees=True)

IK_STEPS_INTERVAL = 2  # Interval in terms of physics_dt steps


def convert_xr_to_world_pose(pose):
    """
        Note: by default we use Local Space reference (see openXR docs) and the unit of measure is meters
        pose is ovrPosef format: right-handed cartisiaan coordinate system,
        flat array of 7 floats as follows:
            1. ovrQuatf: x, y, z, w
            2. ovrVector3f: x, y, z
            (-x is forward, z is left, y is up)
    )
        In isaac world is (+Z up, +X forward), ros is (+Y up, +Z forward) and usd is (+Y up and -Z forward). Defaults to “world”.
        In isaac quaternion is w, x, y, z
    """
    # Old - for OVRLib conversion
    # qx_ovr, qy_ovr, qz_ovr, qw_ovr, px_ovr, py_ovr, pz_ovr = pose
    # orientation = np.array([qw_ovr, qz_ovr, -qx_ovr, qy_ovr])
    # pos = np.array([pz_ovr, -px_ovr, py_ovr])
    new_pose = {}
    new_pose.orientation = np.array(
        [
            pose.orientation.w,  # w
            pose.orientation.z,  # x
            -pose.orientation.x,  # y
            pose.orientation.y,  # z
        ]
    )
    new_pose.position = np.array(
        [
            pose.position.z,
            -pose.position.x,
            pose.position.y,
        ]
    )
    return new_pose


world = World(physics_dt=0.01, rendering_dt=0.02)
world.scene.add_default_ground_plane()

camera_left = world.scene.add(
    Camera(
        prim_path="/World/camera/left",
        name="camera_left",
        position=ORIGIN_POS,
        frequency=CAM_FREQ,
        resolution=CAM_RESOLUTION,
        orientation=ORIGIN_ROT,
    )
)
camera_right = world.scene.add(
    Camera(
        prim_path="/World/camera/right",
        name="camera_right",
        position=ORIGIN_POS + IPD_OFFSET,
        frequency=CAM_FREQ,
        resolution=CAM_RESOLUTION,
        orientation=ORIGIN_ROT,
    )
)

franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
kinematics_solver = KinematicsSolver(franka)
articulation_controller = franka.get_articulation_controller()

world.reset()
camera_left.initialize()
camera_right.initialize()


def main():
    with XrWrapper() as xr_wrapper:
        while True:
            rgb_left = camera_left.get_rgba()[..., :3]
            rgb_right = camera_right.get_rgba()[..., :3]
            xr_wrapper.step(rgb_left, rgb_right)
            world.step(render=True)  # execute one physics step and one rendering step


# # Compute IK
# actions, succ = kinematics_solver.compute_inverse_kinematics(
#     target_position=target_pos,
#     target_orientation=right_ctrl_orientation,
# )
# if succ:
#     articulation_controller.apply_action(actions)
# else:
#     carb.log_warn("IK did not converge to a solution.  No action is being taken.")

# except xr.exception.FormFactorUnavailableError as e:
#     print(e)
#     print("Got error, waiting 5s...")
#     time.sleep(10)
#     pass

if __name__ == "__main__":
    main()
