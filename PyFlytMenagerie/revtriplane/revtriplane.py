"""Implementation of a Revtriplane UAV."""
from __future__ import annotations

import os

import numpy as np
import yaml
from pybullet_utils import bullet_client
from PyFlyt.core.abstractions.base_drone import DroneClass
from PyFlyt.core.abstractions.camera import Camera
from PyFlyt.core.abstractions.gimbals import Gimbals
from PyFlyt.core.abstractions.lifting_surfaces import LiftingSurface, LiftingSurfaces
from PyFlyt.core.abstractions.motors import Motors


class Revtriplane(DroneClass):
    """Revtriplane instance that handles everything about a Revtriplane."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        control_hz: int = 120,
        physics_hz: int = 240,
        drone_model: str = "revtriplane",
        model_dir: None | str = None,
        np_random: None | np.random.RandomState = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 0,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        camera_position_offset: np.ndarray = np.array([-3.0, 0.0, 1.0]),
        starting_velocity: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        """Creates a Revtriplane UAV and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            control_hz (int): control_hz
            physics_hz (int): physics_hz
            drone_model (str): drone_model
            model_dir (None | str): model_dir
            np_random (None | np.random.RandomState): np_random
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            camera_position_offset (np.ndarray): offset position of the camera
            starting_velocity (np.ndarray): vector representing the velocity at spawn
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            control_hz=control_hz,
            physics_hz=physics_hz,
            model_dir=os.path.dirname(__file__),
            drone_model=drone_model,
            np_random=np_random,
        )

        # constants
        self.starting_velocity = starting_velocity

        """Reads revtriplane.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)

            # all lifting surfaces
            surfaces = list()
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=5,
                    command_id=0,
                    command_sign=+1.0,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["left_wing_flapped_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=6,
                    command_id=0,
                    command_sign=-1.0,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["right_wing_flapped_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=3,
                    command_id=1,
                    command_sign=1.0,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["horizontal_tail_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=7,
                    command_id=None,
                    command_sign=+1.0,
                    lifting_unit=np.array([0.0, 0.0, 1.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["main_wing_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=4,
                    command_id=2,
                    command_sign=-1.0,
                    lifting_unit=np.array([0.0, 1.0, 0.0]),
                    forward_unit=np.array([1.0, 0.0, 0.0]),
                    **all_params["vertical_tail_params"],
                )
            )
            self.lifting_surfaces = LiftingSurfaces(lifting_surfaces=surfaces)

            # front motor
            front_motor_params = all_params["front_motor_params"]
            tau = np.array([front_motor_params["tau"]])
            max_rpm = np.array([1.0]) * np.sqrt(
                (front_motor_params["total_thrust"]) / front_motor_params["thrust_coef"]
            )
            thrust_coef = np.array([front_motor_params["thrust_coef"]])
            torque_coef = np.array([front_motor_params["torque_coef"]])
            thrust_unit = np.array([[0.0, 0.0, 1.0]])
            noise_ratio = np.array([front_motor_params["noise_ratio"]])
            self.front_motor = Motors(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                motor_ids=[0],
                tau=tau,
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                thrust_unit=thrust_unit,
                noise_ratio=noise_ratio,
            )

            # rear motors
            motor_ids = [1, 2]
            rear_motor_params = all_params["rear_motor_params"]
            thrust_coef = np.array([rear_motor_params["thrust_coef"]] * 2)
            torque_coef = np.array(
                [
                    -rear_motor_params["torque_coef"],
                    +rear_motor_params["torque_coef"],
                ]
            )
            thrust_unit = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            noise_ratio = np.array([1.0] * 2) * rear_motor_params["noise_ratio"]
            max_rpm = np.array([1.0] * 2) * np.sqrt(
                (rear_motor_params["total_thrust"])
                / (2 * rear_motor_params["thrust_coef"])
            )
            tau = np.array([1.0] * 2) * rear_motor_params["tau"]
            self.rear_motors = Motors(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                motor_ids=motor_ids,
                tau=tau,
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                thrust_unit=thrust_unit,
                noise_ratio=noise_ratio,
            )

            # add the gimbal for front motor
            self.front_motor_gimbal = Gimbals(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                gimbal_unit_1=np.array([[1.0, 0.0, 0.0]]),
                gimbal_unit_2=np.array([[0.0, 1.0, 0.0]]),
                gimbal_tau=np.array([0.01]),
                gimbal_range_degrees=np.array([[0, 90]]),
            )

            # add the gimbal for rear motors
            self.rear_motors_gimbals = Gimbals(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                gimbal_unit_1=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                gimbal_unit_2=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
                gimbal_tau=np.array([0.01, 0.01]),
                gimbal_range_degrees=np.array([[0, 5], [0, 5]]),
            )

        """ CAMERA """
        self.use_camera = use_camera
        if self.use_camera:
            self.camera = Camera(
                p=self.p,
                uav_id=self.Id,
                camera_id=0,
                use_gimbal=use_gimbal,
                camera_FOV_degrees=camera_FOV_degrees,
                camera_angle_degrees=camera_angle_degrees,
                camera_resolution=camera_resolution,
                camera_position_offset=camera_position_offset,
                is_tracking_camera=True,
            )

    def reset(self):
        """Resets the vehicle to the initial state."""
        self.set_mode(0)
        self.setpoint = np.zeros(10)
        self.cmd = np.zeros(10)

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.p.resetBaseVelocity(self.Id, self.starting_velocity, [0, 0, 0])
        self.disable_artificial_damping()
        self.lifting_surfaces.reset()
        self.front_motor.reset()
        self.rear_motors.reset()
        self.front_motor_gimbal.reset()
        self.rear_motors_gimbals.reset()

    def update_control(self):
        """Runs through controllers."""
        # the default mode
        if self.mode == 0:
            self.cmd = self.setpoint.copy()

            # convert pwm to normalised input for lifting surfaces
            self.cmd[0:4] = self.pwm2cmd(
                pwm=np.array(self.setpoint[0:4]),
                pwmrange=np.array([1000, 2000]),
                cmdrange=np.array([-1, 1]),
            )

            # convert pwm to normalised input for front motor gimbal
            self.cmd[6] = self.pwm2cmd(
                pwm=np.array(self.setpoint[6]),
                pwmrange=np.array([1500, 2000]),
                cmdrange=np.array([0, 1]),
            )

            # convert pwm to normalised input for rear motors gimbals
            self.cmd[4:6] = self.pwm2cmd(
                pwm=np.array([self.setpoint[4], self.setpoint[5]]),
                pwmrange=np.array([1000, 2000]),
                cmdrange=np.array([-1, 1]),
            )

            return

        # otherwise, check that we have a custom controller
        if self.mode not in self.registered_controllers.keys():
            raise ValueError(
                f"Don't have other modes aside from 0, received {self.mode}."
            )

        # custom controllers run if any
        self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)

    def update_physics(self):
        """Updates the physics of the vehicle."""
        assert self.cmd[7] >= 0.0, f"thrust `{self.cmd[7]}` must be more than 0.0."

        # move the front motors gimbal
        front_motor_rotation = self.front_motor_gimbal.compute_rotation(
            np.array([0, self.cmd[6]])
        )
        # move the rear motors gimbal
        rear_motors_rotation = self.rear_motors_gimbals.compute_rotation(
            np.array([[0.0, self.cmd[4]], [0.0, self.cmd[5]]])
        )

        # front motor physics
        self.front_motor.physics_update(
            pwm=self.cmd[7], rotation=np.array(front_motor_rotation)
        )
        # rear motors physics
        self.rear_motors.physics_update(
            pwm=np.array([self.cmd[9], self.cmd[8]]),
            rotation=np.array(rear_motors_rotation),
        )

        # lifting surface physics
        self.lifting_surfaces.physics_update(self.cmd)

    def update_state(self):
        """Updates the current state of the UAV.

        This includes: ang_vel, ang_pos, lin_vel, lin_pos.
        """
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        # create the state
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # update all lifting surface velocities
        self.lifting_surfaces.state_update(rotation)

        # update auxiliary information
        self.aux_state = np.concatenate(
            (
                self.lifting_surfaces.get_states(),
                self.front_motor.get_states(),
                self.rear_motors.get_states(),
            )
        )

    def update_last(self):
        """Updates things only at the end of `Aviary.step()`."""
        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()

    def pwm2cmd(self, pwm: np.ndarray, pwmrange: np.ndarray, cmdrange: np.ndarray):
        """Converts PWM input from ardupilot to normalised values in the given range
        pwm: integer input (usually from 1000 to 2000)
        pwmrange: [min, max] range of the pwm input
        cmdrange: [min, max] range to linearly map the pwm input to
        """

        cmd = np.interp(pwm, pwmrange, cmdrange)

        return cmd
