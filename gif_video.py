from moviepy.editor import VideoFileClip
from PIL import Image
import math
import os

def extract_and_grid_frames(video_path, output_png_path, num_frames=16, grid_size=(4, 4)):
    """
    Extracts equally spaced frames from the first second of a video
    and arranges them into a 4x4 grid in a PNG image.

    Args:
        video_path (str): Path to the input MP4 video file.
        output_png_path (str): Path to save the output PNG grid image.
        num_frames (int): The total number of frames to extract (default is 16).
        grid_size (tuple): A tuple (rows, columns) for the grid (default is (4, 4)).
    """

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error loading video file: {e}")
        return

    if clip.duration < 1:
        print("Warning: Video is less than 1 second long. Extracting frames from available duration.")
        duration_to_sample = clip.duration
    else:
        duration_to_sample = 1 # We want frames from the first second

    frame_timestamps = []
    if num_frames > 0:
        # Calculate equally spaced timestamps within the first second
        for i in range(num_frames):
            timestamp = (i / num_frames) * duration_to_sample
            frame_timestamps.append(timestamp)
    else:
        print("Error: num_frames must be greater than 0.")
        clip.close()
        return

    frames = []
    for i, ts in enumerate(frame_timestamps):
        try:
            # Get frame as a PIL Image
            frame_array = clip.get_frame(ts)
            frame_image = Image.fromarray(frame_array)
            frames.append(frame_image)
        except Exception as e:
            print(f"Error extracting frame at {ts} seconds: {e}")
            # If a frame extraction fails, we might still want to proceed with available frames
            # or fill with a blank image, depending on desired behavior.
            # For simplicity, we'll just skip the problematic frame here.

    clip.close()

    if not frames:
        print("No frames were extracted. Cannot create grid.")
        return

    # Determine the size of the grid cells
    first_frame_width, first_frame_height = frames[0].size
    grid_rows, grid_cols = grid_size

    # Calculate the dimensions of the final grid image
    grid_width = grid_cols * first_frame_width
    grid_height = grid_rows * first_frame_height

    # Create a new blank image for the grid
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste each frame into the grid
    for i, frame in enumerate(frames):
        if i >= num_frames: # Stop if we have enough frames for the grid
            break

        row = i // grid_cols
        col = i % grid_cols

        x_offset = col * first_frame_width
        y_offset = row * first_frame_height

        # Resize frame to fit if its dimensions are different (shouldn't happen with get_frame directly)
        if frame.size != (first_frame_width, first_frame_height):
            frame = frame.resize((first_frame_width, first_frame_height))

        grid_image.paste(frame, (x_offset, y_offset))

    # Save the final grid image
    try:
        grid_image.save(output_png_path)
        print(f"Successfully created grid image: {output_png_path}")
    except Exception as e:
        print(f"Error saving grid image: {e}")

if __name__ == "__main__":
    # Example Usage:
    # Replace 'your_video.mp4' with the path to your video file
    # Replace 'output_grid.png' with your desired output file name
    video_file = 'low_stable_x/video/best_model_model.mp4'
    output_image = 'walk.png'

    # Create a dummy video file for testing if you don't have one
    # This part requires opencv-python and numpy to create a simple video
    try:
        import numpy as np
        import cv2

        if not os.path.exists(video_file):
            print(f"Creating a dummy video '{video_file}' for demonstration purposes...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_file, fourcc, 15, (640, 480))

            for i in range(30): # Create a 2-second video at 15 fps
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some text to make frames distinguishable
                cv2.putText(frame, f'Frame {i+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                out.write(frame)
            out.release()
            print("Dummy video created.")

    except ImportError:
        print("Install 'opencv-python' and 'numpy' (pip install opencv-python numpy) to create a dummy video for testing.")
        print("Skipping dummy video creation. Please ensure 'your_video.mp4' exists or create it manually.")
    except Exception as e:
        print(f"Error creating dummy video: {e}")

    # Run the frame extraction and grid creation
    extract_and_grid_frames(video_file, output_image)