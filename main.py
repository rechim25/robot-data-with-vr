from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        # "width": 2800,
        # "height": 2400,
        # "window_width": 2900,
        # "window_height": 2500,
        "headless": True,
        # "multi_gpu": False,
        "renderer": "RayTracedLighting",
        # "denoiser": False,
    }
)

import cProfile
import pstats
import numpy as np
import carb
import time
from xr_wrapper import XrWrapper, Pose
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from omni.isaac.core import World
from omni.isaac.franka import Franka, KinematicsSolver
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

PHYSICS_DT = 0.01
RENDERING_DT = 0.02

CAM_DT = 0.02
CAM_FREQ = 50
CAM_RESOLUTION = (
    int(1832 / 2.3),
    int(1920 / 2.3),
)  # Native Quest 2 resolution is 1832 x 1920 (ratio 1.9)
CAM_ORIGIN_POS = np.array([-1.3, -0.8, 0.5])
CAM_IPD_OFFSET = np.array([0, 0.27, 0])
CAM_ORIGIN_ROT = rot_utils.euler_angles_to_quats(np.array([180, 0, 0]), degrees=True)

EE_ORIGIN_POS = np.array([-1.3, -0.8, 0.5])
EE_ORIGIN_ROT = rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True)

IK_STEPS_INTERVAL = 2  # Interval in terms of physics_dt steps


def log(s) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {str(s)}")


def multiply_np_quats(q1: np.array, q2: np.array) -> np.array:
    r = R.from_quat([q1[1], q1[2], q1[3], q1[0]]) * R.from_quat(
        [q2[1], q2[2], q2[3], q2[0]]
    )
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def update_camera_poses(head_pose: Pose, camera_left, camera_right) -> None:
    # position_target = CAM_ORIGIN_POS + np.array(head_pose.position)
    target_q = multiply_np_quats(np.array(head_pose.orientation), CAM_ORIGIN_ROT)
    camera_left.set_world_pose(orientation=target_q)
    camera_right.set_world_pose(orientation=target_q)


def main() -> None:
    # caleb: phys 60hz
    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    log("Created world")
    world.scene.add_default_ground_plane()
    camera_left = world.scene.add(
        Camera(
            prim_path="/World/camera/left",
            name="camera_left",
            position=CAM_ORIGIN_POS,
            frequency=CAM_FREQ,
            # dt=CAM_DT,
            resolution=CAM_RESOLUTION,
            orientation=CAM_ORIGIN_ROT,
        )
    )
    camera_right = world.scene.add(
        Camera(
            prim_path="/World/camera/right",
            name="camera_right",
            position=CAM_ORIGIN_POS + CAM_IPD_OFFSET,
            frequency=CAM_FREQ,
            # dt=CAM_DT,
            resolution=CAM_RESOLUTION,
            orientation=CAM_ORIGIN_ROT,
        )
    )
    franka = world.scene.add(
        Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
    )
    log("Added Franka arm")
    kinematics_solver = KinematicsSolver(franka)
    articulation_controller = franka.get_articulation_controller()
    log("Created kinematics solver")
    world.reset()
    log("Performed world reset")
    camera_left.initialize()
    camera_right.initialize()
    log("Reset world and initialized cameras")

    with XrWrapper(resolution=CAM_RESOLUTION) as xr_wrapper:
        while True:
            start_time = time.time()
            t = time.time()
            rgb_left = camera_left.get_rgba()[..., :3]
            rgb_right = camera_right.get_rgba()[..., :3]
            log(f"RGBA in {time.time() - t} seconds")

            poses = None
            if rgb_left.ndim == rgb_right.ndim == 3:
                t = time.time()
                poses = xr_wrapper.step(rgb_left, rgb_right)
                log(f"XR step took {time.time() - t} seconds")
            if poses is not None:
                if poses.head:
                    # print(f"Head position: {poses.head.position}")
                    # print(f"Head orientation: {poses.head.position}\n")
                    # update_camera_poses(poses.head, camera_left, camera_right)
                    # position_target = CAM_ORIGIN_POS + np.array(poses.head.position)
                    update_camera_poses(poses.head, camera_left, camera_right)
                if poses.right_hand:
                    pass
                    # print(f"RightHand position: {poses.right_hand.position}")
                    # Compute IK
                    # actions, succ = kinematics_solver.compute_inverse_kinematics(
                    #     target_position=EE_ORIGIN_POS + np.array(poses.right_hand.position),
                    #     target_orientation=EE_ORIGIN_ROT * np.array(poses.right_hand.orientation),
                    # )
                    # if succ:
                    #     articulation_controller.apply_action(actions)
                    # else:
                    #     carb.log_warn(
                    #         "IK did not converge to a solution.  No action is being taken."
                    #     )
            t = time.time()
            world.step(render=True)  # execute one physics step and one rendering step
            log(f"World step took {time.time() - t} seconds")
            log(f"Loop took {time.time() - start_time} seconds\n")


# except xr.exception.FormFactorUnavailableError as e:
#     print(e)
#     print("Got error, waiting 5s...")
#     time.sleep(10)
#     pass

if __name__ == "__main__":
    main()
    simulation_app.close()
