import os
import imageio
import argparse
from stable_baselines3 import PPO

# import your turn‐env and CommandWrapper
from TurnEnv import HexapodTurnEnv
from turnPPO import CommandWrapper


def make_video_for_command(
    model_path: str,
    output_dir: str,
    prefix: str,
    command: float,
    max_steps: int = 500,
    fps: int = 30,
    quality: int = 8,
):
    """
    Generate a single MP4 video for `model_path` performing the specified turn command.

    Args:
        model_path: Path to the .zip model file
        output_dir: Directory to save the video
        prefix: Filename prefix (e.g. 'best_model_right')
        command: Turn command, +1.0 for right, -1.0 for left
        max_steps: Max steps per episode
        fps: Frames per second
        quality: ImageIO quality 1–10
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(model_path)
    model_name = os.path.splitext(base)[0]
    suffix = prefix
    video_name = f"{model_name}_{suffix}.mp4"
    video_path = os.path.join(output_dir, video_name)
    print(f"Creating video {video_name} for command {command} from {base}")

    # setup env and load model
    env = CommandWrapper(HexapodTurnEnv(render=False))
    model = PPO.load(model_path)

    # initialize command and reset
    env.env.current_command = command
    obs, _ = env.reset()

    frames = []
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        # render from base env
        frame = env.env.render(mode="rgb_array")
        frames.append(frame)
        if terminated or truncated:
            break

    env.close()

    # write the video
    imageio.mimwrite(
        video_path,
        frames,
        fps=fps,
        quality=quality,
        macro_block_size=None,
    )
    print(f"Saved: {video_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record two videos (left/right) for the best_model.zip checkpoint."
    )
    parser.add_argument(
        "logs_dir",
        help="Path to your logs directory containing 'best_model/best_model.zip'",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Max steps to roll out per video"
    )
    parser.add_argument(
        "--fps", type=int, default=24,
        help="Frames per second for output videos"
    )
    parser.add_argument(
        "--quality", type=int, default=8,
        help="ImageIO quality (1-10)"
    )
    args = parser.parse_args()

    best_model = os.path.join(args.logs_dir, "best_model", "best_model.zip")
    if not os.path.isfile(best_model):
        print(f"Error: best_model.zip not found at {best_model}")
        exit(1)

    videos_dir = os.path.join(args.logs_dir, "video")
    # generate right-turn video
    make_video_for_command(
        model_path=best_model,
        output_dir=videos_dir,
        prefix="best_model_right",
        command=+1.0,
        max_steps=args.steps,
        fps=args.fps,
        quality=args.quality,
    )
    # generate left-turn video
    make_video_for_command(
        model_path=best_model,
        output_dir=videos_dir,
        prefix="best_model_left",
        command=-1.0,
        max_steps=args.steps,
        fps=args.fps,
        quality=args.quality,
    )
