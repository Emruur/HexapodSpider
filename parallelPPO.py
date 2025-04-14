from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from gymnasium.wrappers import TimeLimit
from HexapodEnv import HexapodEnv
import time

# === Callback, make_env, etc. definitions here ===

def make_env(rank: int):
    def _init():
        env = HexapodEnv(render=False)
        env = TimeLimit(env, max_episode_steps=500)
        env = Monitor(env, filename=f"./logs/train/monitor_{rank}.csv")
        return env
    return _init

class TimestepPrinterCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"👉 Timestep: {self.num_timesteps}")
        return True

if __name__ == "__main__":  # ← ✅ Fix: only spawn processes inside here
    num_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = Monitor(TimeLimit(HexapodEnv(render=False), max_episode_steps=500))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path='./logs/checkpoints/',
        name_prefix='ppo_hexapod'
    )

    timestep_callback = TimestepPrinterCallback(print_freq=1000)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=2,
        tensorboard_log="./logs/tensorboard",
        device="cpu"
    )

    model.learn(
        total_timesteps=3_000_000,
        callback=[eval_callback, checkpoint_callback, timestep_callback]
    )
