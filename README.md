# Hexapod Reinforcement Learning Project

This project focuses on training a hexapod robot to walk forward using reinforcement learning, specifically the Proximal Policy Optimization (PPO) algorithm. It includes the Gymnasium environment definition for the hexapod, the training script, and various utility scripts for visualization, motion capture, and performance analysis.

---

## Project Structure

### Gymnasium Environment

* **`HexapodEnv`**: This class defines the Gymnasium environment for the hexapod, allowing for the simulation and interaction necessary to train the robot to move forward.

### Training Agent

* **`parallelPPO.py`**: This script is responsible for training the hexapod agent within the `HexapodEnv` using a parallelized Proximal Policy Optimization (PPO) algorithm.

---

### Utilities

* **`record_motion.py`**: Loads a trained hexapod model and captures its walking motion in an `mctx` format, which can then be used for controlling a physical hexapod robot.
* **`recordLearning.py`**: Generates a video demonstration of the hexapod walking, loading a previously trained model to showcase its learned behavior.
* **`visualizeAgent.py`**: Launches a PyBullet GUI simulation where you can observe a loaded hexapod model walking in real-time.
* **`plotLearningCurve.py`**: Plots the learning curve from the PPO training process, allowing you to visualize the agent's performance improvement over time.

### Trained Model

The policy of the final trained model is under forward/best_model and a demonstration video is under forward/video/