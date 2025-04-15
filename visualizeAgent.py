from HexapodEnv import HexapodEnv
from stable_baselines3 import PPO

env = HexapodEnv(render=True)
model = PPO.load("reward_x_component/best_model/best_model.zip")

obs, _ = env.reset()

for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
