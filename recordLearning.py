import os
import glob
import imageio
import argparse
from stable_baselines3 import PPO
from HexapodEnv import HexapodEnv


def make_videos_from_checkpoints(
    checkpoint_dir: str,
    output_dir: str,
    prefix: str = "video",
    video_length: int = 500,
    fps: int = 30,
    quality: int = 8,
):
    """
    For each checkpoint ZIP in `checkpoint_dir`, run a rollout in HexapodEnv,
    capture frames, and save them as MP4s in `output_dir`.

    Filenames will be of the form `<prefix>_<suffix>.mp4`, where suffix is
    everything after the first underscore in the zip filename.

    Args:
        checkpoint_dir: path containing .zip model files.
        output_dir: path where videos will be saved.
        prefix: prefix for output filenames (e.g. "video" or "best_model").
        video_length: number of simulation steps per video.
        fps: frames per second for output video.
        quality: quality parameter for imageio (1-10).
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(checkpoint_dir, "*.zip")
    model_paths = sorted(glob.glob(pattern))

    if not model_paths:
        print(f"No checkpoint .zip files found in: {checkpoint_dir}")
        return

    for model_path in model_paths:
        base = os.path.basename(model_path)
        name_no_ext = os.path.splitext(base)[0]
        # extract suffix after first underscore
        suffix = name_no_ext.split("_", 1)[1] if "_" in name_no_ext else name_no_ext
        video_name = f"{prefix}_{suffix}.mp4"
        video_path = os.path.join(output_dir, video_name)
        print(f"Creating {video_name} from {base}")

        env = HexapodEnv(render=False, eval_mode=True)
        model = PPO.load(model_path)
        obs, _ = env.reset()
        frames = []
        for _ in range(video_length):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render(mode="rgb_array")
            frames.append(frame)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

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
        description="Generate MP4 videos from PPO checkpoints and best model in a logs directory."
    )
    parser.add_argument(
        "logs_dir",
        help="Path to the logs directory containing 'checkpoints/' and 'best_model/' subfolders",
    )
    parser.add_argument(
        "--length", type=int, default=500,
        help="Number of steps per video rollout"
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

    # Generate videos for all checkpoints
    checkpoints_dir = os.path.join(args.logs_dir, "checkpoints")
    videos_dir = os.path.join(args.logs_dir, "video")
    make_videos_from_checkpoints(
        checkpoint_dir=checkpoints_dir,
        output_dir=videos_dir,
        prefix="video",
        video_length=args.length,
        fps=args.fps,
        quality=args.quality,
    )

    # Also generate video for best model
    best_model_zip = os.path.join(args.logs_dir, "best_model", "best_model.zip")
    if os.path.isfile(best_model_zip):
        print("\nProcessing best_model.zip...")
        # reuse the same output directory, naming prefix 'best_model'
        make_videos_from_checkpoints(
            checkpoint_dir=os.path.dirname(best_model_zip),
            output_dir=videos_dir,
            prefix="best_model",
            video_length=args.length,
            fps=args.fps,
            quality=args.quality,
        )
    else:
        print(f"No best_model.zip found at {best_model_zip}")
