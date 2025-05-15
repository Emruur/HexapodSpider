import gymnasium as gym
from gymnasium import Env, spaces
import pybullet as p
import pybullet_data
import numpy as np
import os

class HexapodEnv(Env):
    def __init__(self, render=False):
        super(HexapodEnv, self).__init__()
        self.render_mode = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_path = os.path.abspath("pexod.urdf")
        self.time_step = 1.0 / 240.0

        # Define joint limits
        self.max_joint_angle = 1.5
        self.min_joint_angle = -1.5

        # Placeholder for joints
        self.num_joints = None
        self.joint_indices = []

        # Temporary dummy spaces (will be updated in reset)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset()  # triggers loading and sets real action/obs space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.2]
        self.robot = p.loadURDF(self.urdf_path, start_pos, useFixedBase=False)

        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = [
            i for i in range(self.num_joints)
            if p.getJointInfo(self.robot, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
        ]

        self.initial_pos = np.array(start_pos)

        for j in self.joint_indices:
            p.resetJointState(self.robot, j, 0)

        # Set spaces dynamically
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_joint_angle,
            high=self.max_joint_angle,
            shape=(len(self.joint_indices),),
            dtype=np.float32
        )
        
        # ✅ Initialize jitter tracking
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_linear, base_angular = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_ori)

        self.prev_base_ori = [roll, pitch, yaw]
        self.prev_base_linear = base_linear
        self.prev_base_angular = base_angular

        return obs, {}

    def step(self, action):
        # --- Apply the action ---
        action = np.clip(action, self.min_joint_angle, self.max_joint_angle)
        max_joint_velocity = 1.0

        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=action[i],
                maxVelocity=max_joint_velocity,
                force=30
            )

        for _ in range(30):
            p.stepSimulation()

        # --- Get pose and velocities ---
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_linear, base_angular = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(base_ori)

        # --- Truncation condition: flipped over ---
        upside_down = abs(roll) > 2.0 or abs(pitch) > 2.0
        terminated = False
        truncated = upside_down

        # --- Components for reward ---
        forward = base_linear[0]
        lateral = base_linear[1]
        height = base_pos[2]

        # --- Jitter components ---
        if self.prev_base_ori is None:
            self.prev_base_ori = [roll, pitch, yaw]
            self.prev_base_linear = base_linear
            self.prev_base_angular = base_angular

        prev_roll, prev_pitch, prev_yaw = self.prev_base_ori

        orientation_jitter = abs(roll - prev_roll) + abs(pitch - prev_pitch) + abs(yaw - prev_yaw)
        velocity_jitter = np.linalg.norm(np.array(base_linear) - np.array(self.prev_base_linear))
        angular_jitter = np.linalg.norm(np.array(base_angular) - np.array(self.prev_base_angular))

        # --- Reward calculation ---
        reward = 0.0
        reward += forward                             # ✅ Forward movement
        reward -= 0.2 * abs(lateral)                  # ⛔ Sideways drift
        reward -= 0.1 * abs(yaw)                      # ⛔ Rotational deviation
        reward -= 0.3 * abs(roll)                     # ⛔ Body tilt
        reward -= 0.3 * abs(pitch)
        reward -= 2.0 * abs(height - 0.1)             # ⛔ Bouncing

        reward -= 0.5 * orientation_jitter            # ⛔ Twitching rotation
        reward -= 0.1 * velocity_jitter               # ⛔ Sudden shifts
        reward -= 0.1 * angular_jitter                # ⛔ Wobbling

        # --- Zero reward if flipped ---
        if truncated:
            reward = 0.0
            
            
        track_error = abs(base_pos[1])  # how far it drifted from y = 0
        reward -= 0.1 * track_error

        # --- Store current state for next step ---
        self.prev_base_ori = [roll, pitch, yaw]
        self.prev_base_linear = base_linear
        self.prev_base_angular = base_angular

        # --- Final return ---
        obs = self._get_obs()
        info = {
            "forward_vel": forward,
            "lateral_vel": lateral,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "height": height,
            "reward": reward,
            "orientation_jitter": orientation_jitter,
            "velocity_jitter": velocity_jitter,
            "angular_jitter": angular_jitter
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        joint_positions = np.array([s[0] for s in joint_states], dtype=np.float32)
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        base_linear, base_angular = p.getBaseVelocity(self.robot)
        obs = np.concatenate([joint_positions, base_ori, base_linear, base_angular])
        return obs.astype(np.float32)  # ✅ enforce dtype here

    def render(self, mode="human", width=640, height=480):
        """
        Render the scene.
        
        Args:
            mode (str): "human" to move the GUI camera; 
                        "rgb_array" to return an image array.
            width (int): width of returned image (rgb_array mode).
            height (int): height of returned image.
        
        Returns:
            np.ndarray or None: If mode=="rgb_array", returns an HxWx3 uint8 array.
                                Otherwise returns None.
        """
        # Always get the robot’s current base position as camera target
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        
        # Parameters for our virtual camera:
        cam_distance = 1.5
        cam_yaw = 45
        cam_pitch = -30
        
        # In human mode: just move the debug camera
        if mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=cam_distance,
                cameraYaw=cam_yaw,
                cameraPitch=cam_pitch,
                cameraTargetPosition=base_pos
            )
            return None
        
        # In rgb_array mode: compute view & projection matrices and grab an image
        if mode == "rgb_array":
            # Build view matrix (where the camera is, where it looks, and up-vector)
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[
                    base_pos[0] + cam_distance * np.cos(np.deg2rad(cam_yaw)),
                    base_pos[1] + cam_distance * np.sin(np.deg2rad(cam_yaw)),
                    base_pos[2] + cam_distance * np.sin(np.deg2rad(-cam_pitch))
                ],
                cameraTargetPosition=base_pos,
                cameraUpVector=[0, 0, 1]
            )
            
            # Build projection matrix
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,                # field of view
                aspect=float(width)/height,
                nearVal=0.1,
                farVal=100.0
            )
            
            # Grab the image
            _, _, px, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # px is a flat array of RGBA; reshape and drop alpha
            img = np.reshape(px, (height, width, 4))[:, :, :3]
            return img
        
        # If somebody passes a totally unsupported mode:
        raise ValueError(f"Unsupported render mode: {mode}")


    def close(self):
        p.disconnect(self.physics_client)
