from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from HexapodEnv import HexapodEnv  # Your custom env
from gymnasium.wrappers import TimeLimit



env = Monitor(TimeLimit(HexapodEnv(render=False), max_episode_steps=500), filename="./logs/train/monitor.csv")
eval_env = Monitor(TimeLimit(HexapodEnv(render=False), max_episode_steps=500))
check_env(env, warn= True)




class TimestepPrinterCallback(BaseCallback):
    def __init__(self, print_freq=100, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"👉 Timestep: {self.num_timesteps}")
        return True


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
    verbose=2,  # show more details
    tensorboard_log="./logs/tensorboard",
    device="auto"
)

model.learn(
    total_timesteps=100_000,
    callback=[eval_callback, checkpoint_callback,timestep_callback]
)
