import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from TurnEnv import HexapodTurnEnv

class CommandWrapper(gym.Wrapper):
    def __init__(self, env, command: float = -1.0):
        super().__init__(env)
        self.env.current_command = float(command)
        orig = env.observation_space
        low  = np.concatenate(([-1.0], orig.low),  axis=0)
        high = np.concatenate(([ 1.0], orig.high), axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        cmd = np.array([self.env.current_command], dtype=np.float32)
        return np.concatenate((cmd, obs)), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        cmd = np.array([self.env.current_command], dtype=np.float32)
        return np.concatenate((cmd, obs)), reward, done, truncated, info

class TimestepPrinterCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"👉 Timestep: {self.num_timesteps}")
        return True

def make_env(rank: int, base_log_dir: str, command: float):
    def _init():
        env = HexapodTurnEnv(render=False)
        env = CommandWrapper(env, command=command)
        env = TimeLimit(env, max_episode_steps=500)
        env = Monitor(env, filename=f"{base_log_dir}/monitor_{rank}.csv")
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir", "-L",
        type=str,
        default="./one_way_-1_strict",
        help="Base directory for all logging (monitor, tensorboard, checkpoints, etc.)"
    )
    parser.add_argument(
        "--command", "-C",
        type=float,
        default=-1.0,
        help="Fixed turn command to prepend to each observation"
    )
    parser.add_argument(
        "--load-model-path", "-M",
        type=str,
        default=None,
        help="Optional: path to an existing model.zip to load (won’t overwrite save-dir)"
    )
    args = parser.parse_args()

    num_envs  = 16
    train_log = f"{args.log_dir}/train"
    eval_log  = f"{args.log_dir}/eval"

    # vectorized envs
    train_vec = SubprocVecEnv([make_env(i, train_log, args.command) for i in range(num_envs)])
    eval_vec  = SubprocVecEnv([make_env(0, eval_log, args.command)])

    # load or create model
    if args.load_model_path:
        model = PPO.load(args.load_model_path, env=train_vec, device="cpu")
        print(f"🔄 Loaded model from {args.load_model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            train_vec,
            verbose=2,
            tensorboard_log=f"{args.log_dir}/tensorboard",
            device="cpu",
        )

    # callbacks
    eval_callback = EvalCallback(
        eval_env=eval_vec,
        best_model_save_path=f"{args.log_dir}/best_model",
        log_path=eval_log,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // num_envs,
        save_path=f"{args.log_dir}/checkpoints",
        name_prefix='ppo_hexapod_turn'
    )
    timestep_callback = TimestepPrinterCallback(print_freq=1_000)

    # train
    model.learn(
        total_timesteps=30_000_000,
        callback=[eval_callback, checkpoint_callback, timestep_callback],
    )
