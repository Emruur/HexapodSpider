# Hexapod Reinforcement Learning Project

This project focuses on training a hexapod robot to walk forward using reinforcement learning, specifically the Proximal Policy Optimization (PPO) algorithm. It includes the Gymnasium environment definition for the hexapod, the training script, and various utility scripts for visualization, motion capture, and performance analysis.

---

## How to Use It

1. Plug the **BT-410 Dongle** into your laptop and the **BT-410 Slave** into the BIOLOID King Spider.
2. Power on the robot.
3. Switch to **Play mode** and press the **Start** button.
4. Run `main.py` on the laptop.  
   >  Requires downloading the LLM model **Qwen1.5-32B-Chat** from Hugging Face.
5. When the terminal displays “Please speak”, speak clearly in front of your laptop.

## Setup Instructions (Environment Setup)

1. Clone this repository:
   ```bash
   git clone https://github.com/Emruur/HexapodSpider.git
   cd HexapodSpider
2. Create and open a virtual environment:
   ```bash
   conda create --name SPIRAL python=3.10
   conda activate SPIRAL
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
4. Download the LLM model (Qwen/Qwen1.5-32B-Chat) from Hugging Face:

   Visit: https://huggingface.co/Qwen/Qwen1.5-32B-Chat

   Agree to the terms and manually download all model files.

   Create a folder in your project root named Qwen-Qwen1.5-32B-Chat and place all the files inside.

   The project will load the model from that folder directly.



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

### BOILOID motion and task files

* **`spiral_motion.mtnx`**: Motion file for R+ Motion 2, containing both default motions and those trained via reinforcement learning.
* **`spiral_task.tskx`**: Task file for R+ Task 2 that maps buttons to corresponding motions.

### Speech Control Pipeline

* **`main.py`**: Implements the speech control pipeline, including speech-to-text conversion, LLM-based command decoding, RC-100 button simulation, and wireless robot control.
