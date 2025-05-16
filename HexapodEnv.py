import gymnasium as gym
from gymnasium import Env, spaces
import pybullet as p
import pybullet_data
import numpy as np
import os

class HexapodEnv(Env):
    def __init__(self, render=False, max_episodes=100000):
        super(HexapodEnv, self).__init__()
        self.render_mode = render
        # --- Physics client for space setup ---
        setup_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdf_path = os.path.abspath("pexod.urdf")
        self.time_step = 1.0 / 240.0
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")
        temp_robot = p.loadURDF(self.urdf_path, [0, 0, 0.2], useFixedBase=False)
        # identify movable joints
        all_joints = p.getNumJoints(temp_robot)
        self.joint_indices = [i for i in range(all_joints)
                              if p.getJointInfo(temp_robot, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        # reset joint states
        for j in self.joint_indices:
            p.resetJointState(temp_robot, j, 0)
        # sample dummy state
        obs = self._get_obs_for_robot(temp_robot)
        # define action/obs spaces
        self.min_joint_angle = -1.5
        self.max_joint_angle =  1.5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=obs.shape, dtype=np.float32)
        self.action_space      = spaces.Box(low=self.min_joint_angle, high=self.max_joint_angle,
                                            shape=(len(self.joint_indices),), dtype=np.float32)
        # cleanup setup sim
        p.disconnect(setup_client)
        # --- Real physics client ---
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Curriculum
        self.turn_command       = 0
        self.reference_heading  = 0.0
        self.episode_count      = 0
        self.max_episodes       = max_episodes
        self.straight_prob      = 1.0
        self.turn_prob          = 0.0
        self.mid_change_prob    = 0.0

        # Persist placeholder
        self.num_joints = None
        self.robot       = None

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, [0,0,0.2], useFixedBase=False)
        self.num_joints = p.getNumJoints(self.robot)
        for j in self.joint_indices:
            p.resetJointState(self.robot, j, 0)
        # curriculum update
        self._update_curriculum()
        self._manage_turn_command('reset')
        # tracking
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot)
        lin, ang = p.getBaseVelocity(self.robot)
        # convert to arrays for arithmetic
        self.prev_base_linear  = np.array(lin, dtype=np.float32)
        self.prev_base_angular = np.array(ang, dtype=np.float32)
        self.prev_roll, self.prev_pitch, self.prev_yaw = p.getEulerFromQuaternion(base_ori)
        self.prev_desired_yaw_rate = 0.0
        self.reference_heading      = self.prev_yaw
        return self._get_obs(), {}

    def step(self, action, turn_command=None):
        old_cmd = self.turn_command
        if turn_command is not None:
            self.turn_command = int(turn_command)
        else:
            self._manage_turn_command('step')
        # apply actions
        p_action = np.clip(action, self.min_joint_angle, self.max_joint_angle)
        for i, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=p_action[i], maxVelocity=1.0, force=30)
        for _ in range(30): p.stepSimulation()
        # kinematics
        pos, ori       = p.getBasePositionAndOrientation(self.robot)
        lin, ang       = p.getBaseVelocity(self.robot)
        # convert to arrays
        lin_vel         = np.array(lin, dtype=np.float32)
        ang_vel         = np.array(ang, dtype=np.float32)
        roll, pitch, yaw = p.getEulerFromQuaternion(ori)
        if self.turn_command != old_cmd:
            self.reference_heading = yaw
        truncated = abs(roll)>2.0 or abs(pitch)>2.0
        desired_yaw_rate = self.turn_command * 0.5
        # forward relative to reference heading
        ref_vec = np.array([np.cos(self.reference_heading), np.sin(self.reference_heading), 0.], dtype=np.float32)
        forward = float(np.dot(lin_vel, ref_vec))
        # turn bonus
        yaw_err    = abs(ang_vel[2] - desired_yaw_rate)
        turn_bonus = float(np.exp(-5.0 * yaw_err))
        # lateral penalty
        lat_pen = 0.2 * abs(np.dot(lin_vel, np.array([-ref_vec[1], ref_vec[0], 0.])))
        # stability
        lin_j     = np.linalg.norm(lin_vel - self.prev_base_linear)
        stability = 0.3*(abs(roll)+abs(pitch)) + 2.0*abs(pos[2]-0.1) + 0.1*lin_j
        # yaw accel penalty
        actual_acc = (ang_vel[2] - self.prev_base_angular[2]) / self.time_step
        expected_acc = (desired_yaw_rate - self.prev_desired_yaw_rate) / self.time_step
        yaw_pen     = 0.1 * abs(actual_acc - expected_acc)
        reward = 0.0 if truncated else float(forward + turn_bonus - (lat_pen + stability + yaw_pen))
        obs = self._get_obs()
        info = {'command': self.turn_command,
                'reward_terms': {
                    'forward': forward,
                    'turn_bonus': turn_bonus,
                    'lat_pen': lat_pen,
                    'stability': stability,
                    'yaw_pen': yaw_pen
                }}
        # save for next step
        self.prev_base_linear        = lin_vel
        self.prev_base_angular       = ang_vel
        self.prev_roll, self.prev_pitch, self.prev_yaw = roll, pitch, yaw
        self.prev_desired_yaw_rate   = desired_yaw_rate
        return obs, reward, False, truncated, info

    def _get_obs_for_robot(self, robot):
        js = p.getJointStates(robot, self.joint_indices)
        jp = np.array([s[0] for s in js], dtype=np.float32)
        pos, ori = p.getBasePositionAndOrientation(robot)
        lin, ang  = p.getBaseVelocity(robot)
        return np.concatenate([jp, ori, lin, ang, [0.0]]).astype(np.float32)

    def _get_obs(self):
        return self._get_obs_for_robot(self.robot)

    def _update_curriculum(self):
        phase = min(self.episode_count/self.max_episodes, 1.0)
        self.straight_prob   = 1.0 - phase
        self.turn_prob       = phase / 2.0
        self.mid_change_prob = 0.1 * phase
        self.episode_count  += 1

    def _manage_turn_command(self, site):
        probs   = [self.turn_prob, self.straight_prob, self.turn_prob]
        choices = [-1, 0, 1]
        if site == 'reset':
            self.turn_command = int(np.random.choice(choices, p=probs))
        elif site == 'step' and np.random.rand() < self.mid_change_prob:
            self.turn_command = int(np.random.choice(choices, p=probs))

    def render(self, mode="human", width=640, height=480):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        if mode == "human":
            p.resetDebugVisualizerCamera(1.5, 45, -30, pos)
            return
        view = p.computeViewMatrix(
            cameraEyePosition=[
                pos[0] + 1.5*np.cos(np.deg2rad(45)),
                pos[1] + 1.5*np.sin(np.deg2rad(45)),
                pos[2] + 1.5*np.sin(np.deg2rad(-30))
            ],
            cameraTargetPosition=pos,
            cameraUpVector=[0,0,1]
        )
        proj = p.computeProjectionMatrixFOV(60, float(width)/height, 0.1, 100)
        _, _, px, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return np.reshape(px, (height, width, 4))[:, :, :3]

    def close(self):
        p.disconnect(self.physics_client)
