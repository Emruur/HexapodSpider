import gymnasium as gym
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from TurnEnv import HexapodTurnEnv


tf = time = None  # silence linter if unused

class CommandWrapper(gym.Wrapper):
    """
    Wraps the hexapod env to prepend a constant turn command (±1) to each observation,
    and provides a flip_command() method for external toggling.
    Also adjusts the observation_space accordingly.
    """
    def __init__(self, env):
        super().__init__(env)
        # initialize command
        self.env.current_command = 1.0
        # adjust observation_space: prepend one entry for command
        orig_space = env.observation_space
        low = np.concatenate(([-1.0], orig_space.low), axis=0)
        high = np.concatenate(([1.0], orig_space.high), axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # action_space remains unchanged
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # prepend current command
        cmd = np.array([self.env.current_command], dtype=np.float32)
        return np.concatenate((cmd, obs)), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        cmd = np.array([self.env.current_command], dtype=np.float32)
        return np.concatenate((cmd, obs)), reward, done, truncated, info

    def flip_command(self):
        """Flip the base command (+1 ↔ -1)."""
        self.env.current_command *= -1.0

class PlateauFlipCallback(EvalCallback):
    """
    Flips the turn command in all envs when performance plateaus.

    Args:
        plateau_window (int): number of eval points to track
        min_delta (float): minimum improvement to avoid plateau
    """
    def __init__(self,
                 eval_env,
                 best_model_save_path: str,
                 log_path: str,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 5,
                 plateau_window: int = 5,
                 min_delta: float = 0.01,
                 **kwargs):
        super().__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            **kwargs
        )
        self.plateau_window = plateau_window
        self.min_delta = min_delta
        self.window = deque(maxlen=plateau_window)

    def _on_eval_end(self) -> bool:
        result = super()._on_eval_end()
        mean_reward = float(self.last_mean_reward)
        if len(self.window) == self.window.maxlen:
            if mean_reward - self.window[0] < self.min_delta:
                print("🔄 Plateau detected, flipping commands")
                # flip commands in all wrapped envs
                self.training_env.env_method('flip_command')
                self.eval_env.env_method('flip_command')
                self.window.clear()
        self.window.append(mean_reward)
        return result

class TimestepPrinterCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"👉 Timestep: {self.num_timesteps}")
        return True


def make_env(rank: int, log_dir: str):
    def _init():
        base = HexapodTurnEnv(render=False)
        wrapped = CommandWrapper(base)
        wrapped = TimeLimit(wrapped, max_episode_steps=500)
        wrapped = Monitor(wrapped, filename=f"{log_dir}/monitor_{rank}.csv")
        return wrapped
    return _init

if __name__ == "__main__":
    num_envs = 16
    train_log = "./logs/train"
    eval_log  = "./logs/eval"

    # training environments
    train_vec = SubprocVecEnv([make_env(i, train_log) for i in range(num_envs)])
    
    # evaluation environment
    eval_vec = SubprocVecEnv([make_env(0, eval_log)])

    model = PPO(
        "MlpPolicy",
        train_vec,
        verbose=2,
        tensorboard_log="./logs/tensorboard",
        device="cpu",
    )

    plateau_callback = PlateauFlipCallback(
        eval_vec,
        best_model_save_path="./logs/best_model",
        log_path=eval_log,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        plateau_window=5,
        min_delta=0.01
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // num_envs,
        save_path='./logs/checkpoints',
        name_prefix='ppo_hexapod_turn'
    )

    timestep_callback = TimestepPrinterCallback(print_freq=1_000)
    
    model = PPO.load("./turn_left/best_model/best_model.zip", env=train_vec, device="cpu")
    model.learn(
        total_timesteps=30_000_000,
        callback=[plateau_callback, checkpoint_callback, timestep_callback],
    )
