import numpy as np
from HexapodEnv import HexapodEnv
from stable_baselines3 import PPO
import xml.etree.ElementTree as ET

def rad_to_deg(radians):
    return np.degrees(radians).tolist()

def format_pose(deg_angles):
    return " ".join([f"{angle:.2f}" for angle in deg_angles])

def record_motion(model_path, num_frames=4, frame_interval=120):
    env = HexapodEnv(render=False)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    recorded_poses = []

    for i in range(num_frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        joint_angles_deg = rad_to_deg(action)
        recorded_poses.append((frame_interval * (i + 1), joint_angles_deg))

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return recorded_poses

def create_mctx_xml(poses, name="RL Walk", accel="8", softness="5 " * 18):
    page = ET.Element("Page", name=name)
    param = ET.SubElement(page, "param", compileSize="1", acceleration=accel, softness=softness.strip())
    steps = ET.SubElement(page, "steps")

    for frame, angles in poses:
        pose_str = format_pose(angles)
        ET.SubElement(steps, "step", frame=str(frame), pose=pose_str)

    return ET.tostring(page, encoding='unicode')

if __name__ == "__main__":
    model_path = "reward_no_jitter/best_model/best_model.zip"
    poses = record_motion(model_path, num_frames=4, frame_interval=120)
    xml_string = create_mctx_xml(poses, name="RL Walk")

    with open("ppo_walk.mctx", "w") as f:
        f.write(xml_string)

    print("✅ Motion recorded and saved to 'ppo_walk.mctx'")
import numpy as np
from HexapodEnv import HexapodEnv
from stable_baselines3 import PPO
import xml.etree.ElementTree as ET

def rad_to_deg(radians):
    return np.degrees(radians).tolist()

def format_pose(deg_angles):
    return " ".join([f"{angle:.2f}" for angle in deg_angles])

def record_motion(model_path, num_frames=4, frame_interval=120):
    env = HexapodEnv(render=False)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    recorded_poses = []

    for i in range(num_frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        joint_angles_deg = rad_to_deg(action)
        recorded_poses.append((frame_interval * (i + 1), joint_angles_deg))

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return recorded_poses

def create_mctx_xml(poses, name="RL Walk", accel="8", softness="5 " * 18):
    page = ET.Element("Page", name=name)
    param = ET.SubElement(page, "param", compileSize="1", acceleration=accel, softness=softness.strip())
    steps = ET.SubElement(page, "steps")

    for frame, angles in poses:
        pose_str = format_pose(angles)
        ET.SubElement(steps, "step", frame=str(frame), pose=pose_str)

    return ET.tostring(page, encoding='unicode')

if __name__ == "__main__":
    model_path = "reward_no_jitter/best_model/best_model.zip"
    poses = record_motion(model_path, num_frames=10, frame_interval=30)
    xml_string = create_mctx_xml(poses, name="RL Walk")

    with open("ppo_walk.mctx", "w") as f:
        f.write(xml_string)

    print("✅ Motion recorded and saved to 'ppo_walk.mctx'")
