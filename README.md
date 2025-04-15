## Related Work

### Hierarchical RL and terrain planning

https://www.mdpi.com/2076-0825/12/2/75#:~:text=Gait%20plays%20a%20decisive%20role,superior%20reinforcement%20learning%20algorithms%20can

hierarchical reinforcement learning to achieve stability on complex terrains, using a two-level controller to handle planning and reflexes for a hexapod

### Imitation-Enhanced RL

https://acris.aalto.fi/ws/portalfiles/portal/169655192/24-2152_04_MS.pdf


### CPG RL enhancing
https://www.mdpi.com/2076-0825/12/4/157#:~:text=applicability,to%20the%20traditional%20CPG%20method


Huang et al. (2024) similarly showed that augmenting a standard CPG gait with a learned correction policy improved learning efficiency and terrain flexibility, outperforming either method in isolation ￼ ￼.










In summary, traditional methods like CPGs and PID-controlled gait engines provide a strong starting point and were the state-of-the-art for decades in hexapod locomotion. They produce smooth, repeatable gaits but lack adaptability. RL-based approaches, on the other hand, adapt gait strategies automatically to maximize performance criteria (speed, efficiency, stability) across varied environments, at the cost of requiring intensive training and careful deployment. Notably, the best results in recent research often combine the two – using the reliability of classical methods together with the flexibility of learning. For instance, a hexapod might use a CPG to ensure rhythmic leg movements and an RL policy to adjust the amplitude or phase of those movements in response to sensed terrain ￼ ￼. This way, one can get the best of both worlds: the safety of known stable patterns and the adaptability of learned behaviors.


### Legged GYM

https://github.com/leggedrobotics/legged_gym

### Domain Randomization

Adaptive Gait Generation for Hexapod Robots Based on Reinforcement Learning and Hierarchical Framework

### Actuators and Joints
This means the policy must be integrated with the robot’s low-level controller. A common solution is to have the RL policy output desired joint angles (or angle offsets) each time step, which the servo controllers then try to achieve.



# IMPORTANT

Real hexapods are usually equipped with an IMU (inertial measurement unit) to sense body roll/pitch/yaw angles and accelerations. They also read joint encoders for angles and possibly motor currents (which can proxy for torque or detecting contact when a leg encounters resistance). If the RL policy was trained with perfect state information (e.g., exact orientation), the real robot needs to provide that – meaning the IMU data must be fused and filtered to give a clean orientation estimate. 




### Some nice URDFS

https://github.com/chrismailer/hexapod-sim