import gymnasium as gym
from gymnasium import Env, spaces
import pybullet as p
import pybullet_data
import numpy as np
import os

class HexapodTurnEnv(Env):
    """
    Gymnasium environment for stationary hexapod turning.
    The turn command (±1) is set externally via env.current_command.
    Policy outputs joint-angle targets; reward encourages fast, smooth, jitter-free turning
    around yaw axis while maintaining height, level body, and stationary XY position.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, render=False, desired_yaw_rate=0.5):
        super().__init__()
        self.render_mode = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_path = os.path.abspath("pexod.urdf")
        self.time_step = 1.0 / 240.0
        self.desired_yaw_rate = desired_yaw_rate

        # joint limits
        self.max_joint_angle = 1.5
        self.min_joint_angle = -1.5

        # placeholders to be set in reset
        self.robot = None
        self.joint_indices = []
        # external command (±1)
        self.current_command = 1.0

        # track previous angular velocity (for smoothness)
        self.prev_yaw_rate = 0.0

        # initial reset to define spaces
        raw_obs, _ = self.reset()
        obs_dim = raw_obs.shape[0]
        act_dim = len(self.joint_indices)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_joint_angle, high=self.max_joint_angle,
                                       shape=(act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # default command if not set externally
        if not hasattr(self, 'current_command'):
            self.current_command = 1.0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.2]
        self.robot = p.loadURDF(self.urdf_path, start_pos, useFixedBase=False)

        # identify controllable joints
        n_joints = p.getNumJoints(self.robot)
        self.joint_indices = [i for i in range(n_joints)
                              if p.getJointInfo(self.robot, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]
        # reset joint positions
        for j in self.joint_indices:
            p.resetJointState(self.robot, j, 0)

        # reset smoothness tracker
        self.prev_yaw_rate = 0.0

        raw_obs = self._get_raw_obs()
        return raw_obs, {}

    def step(self, action):
        # clip action
        action = np.clip(action, self.min_joint_angle, self.max_joint_angle)
        # apply controls
        for idx, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(bodyIndex=self.robot,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=float(action[idx]),
                                    maxVelocity=1.0,
                                    force=30)
        # simulate
        for _ in range(30):
            p.stepSimulation()

        # get state
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_ori)
        yaw_rate = angular_vel[2]

        # Reward terms simplified for better learning
        # 1) turning in correct direction: reward proportional to yaw_rate * command
        r_turn = yaw_rate * self.current_command
        # 2) moderate encouragement of turning speed magnitude
        r_speed = 0.5 * abs(yaw_rate)
        # 3) small penalty for height deviation
        r_height = -2.0 * abs(base_pos[2] - 0.15)
        # 4) small penalty for tilt
        r_tilt = -1.0 * (abs(roll) + abs(pitch))
        # 5) small penalty for XY drift
        r_center = -2.0 * (abs(base_pos[0]) + abs(base_pos[1]))
        # 6) smoothness: small penalty for yaw acceleration
        r_smooth = -2 * abs(yaw_rate - self.prev_yaw_rate)

        reward = r_turn + r_speed + r_height + r_tilt + r_center + r_smooth

        # termination: flipped or drifted too far
        truncated = False
        if abs(roll) > 0.5 or abs(pitch) > 0.5:
            truncated = True
            reward = -10.0

        # update for smoothness
        self.prev_yaw_rate = yaw_rate

        raw_obs = self._get_raw_obs()
        info = {
            "yaw_rate": yaw_rate,
            "height": base_pos[2],
            "roll": roll,
            "pitch": pitch,
            "xy_pos": base_pos[:2]
        }
        return raw_obs, reward, False, truncated, info

    def _get_raw_obs(self):
        states = p.getJointStates(self.robot, self.joint_indices)
        joint_pos = np.array([s[0] for s in states], dtype=np.float32)
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        return np.concatenate([joint_pos, base_ori, linear_vel, angular_vel]).astype(np.float32)

    def render(self, mode="human", width=640, height=480):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        cam_dist, cam_yaw, cam_pitch = 1.5, 45, -30
        if mode == "human":
            p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, base_pos)
            return None
        elif mode == "rgb_array":
            view = p.computeViewMatrix(cameraEyePosition=[
                base_pos[0] + cam_dist * np.cos(np.deg2rad(cam_yaw)),
                base_pos[1] + cam_dist * np.sin(np.deg2rad(cam_yaw)),
                base_pos[2] + cam_dist * np.sin(np.deg2rad(-cam_pitch))],
                cameraTargetPosition=base_pos,
                cameraUpVector=[0, 0, 1])
            proj = p.computeProjectionMatrixFOV(60, float(width)/height, 0.1, 100.0)
            _, _, px, _, _ = p.getCameraImage(width, height, view, proj,
                                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
            img = np.reshape(px, (height, width, 4))[:, :, :3]
            return img
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        p.disconnect(self.physics_client)
