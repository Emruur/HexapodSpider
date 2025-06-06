# Hexapod Reinforcement Learning and Speech Control Project

This project focuses on enabling a BIOLOID King Spider (hexapod) robot to perform intelligent motion using reinforcement learning and flexible voice command control.

It consists of two main components:

1. Reinforcement Learning for Locomotion

   The robot is trained to walk forward using the Proximal Policy Optimization (PPO) algorithm. This part includes:

* A custom Gymnasium environment modeling the hexapod robot using URDF

* A training script for learning walking policies

* Utilities for visualizing training results and capturing motion sequences

* Code to transfer the learned policy from simulation to the physical robot

2. Speech-Controlled Command Execution

   Voice commands are used to control the robot in real time. This is achieved by:

* Capturing speech using a microphone and converting it to text

* Decoding user commands using a local Large Language Model (LLM), Qwen1.5-32B-Chat

* Mapping commands to predefined or learned robot actions (e.g., walk, turn, wave)

* Sending control signals wirelessly to the robot using RC-100 packet simulation via BT-410

This project demonstrates a complete pipeline from simulation-based learning to natural-language interaction with a physical robot.

![strength](https://github.com/user-attachments/assets/ff70c476-ed5d-4cf8-b8dc-b18d3ac606bf)


Video Example:

* Simple Command Execution: https://drive.google.com/file/d/1tkVYsKviaVzhwGa_GgEwMXZ9wAY_NEof/view?usp=sharing
* Complex Command Execution: https://drive.google.com/file/d/1D2H8iQ8qmOL1wgvzWmyWGQ6vhA2BpBJo/view?usp=sharing
* Creative Command (Attack and Sit): https://drive.google.com/file/d/1DN0JRS9mZ80HVuQgIOmWm1Qswg_qyBHI/view?usp=sharing
* Reinforcement Learning Motion in Real Robot: https://drive.google.com/file/d/1ERcQ1-yFYoEH67KoNyCswyEVPRcAYA-p/view?usp=sharing
* Simulated Forward Walking Behavior: https://drive.google.com/file/d/12A9_CQv3DT8EOlskUJWT63uXcT6Ja75a/view?usp=sharing
* Simulated Turning Left Behavior: https://drive.google.com/file/d/1dRnd5HAx9iuXx3ri23E7YNaPQ9Su43i1/view?usp=sharing
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

## Changing the Communication Port

If you're using a different serial device (e.g., on Windows or another machine), you need to update the communication port in the code.

In `main`, locate the following line:

   ```python
   ser = serial.Serial('/dev/tty.usbserial-AB0MI2NT', baudrate=57600, timeout=1)
   ```

## How to Find Your Port

You can use this Python snippet to list all available serial ports:

   ```python
   import serial.tools.list_ports
   print([port.device for port in serial.tools.list_ports.comports()])
   ```


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
