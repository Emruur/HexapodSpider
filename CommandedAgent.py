import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from TurnEnv import HexapodTurnEnv
import math

# — load your three models —
model_fwd   = PPO.load("low_stable_x/best_model/best_model.zip",   device="cpu")
model_left  = PPO.load("one_way_1_no_load/best_model/best_model.zip",      device="cpu")
model_right = PPO.load("one_way_-1_no_load/best_model/best_model.zip",     device="cpu")

env = HexapodTurnEnv(render=True)
raw_obs, _ = env.reset()

# state variables
mode            = "forward"     # one of "forward","left","right"
current_model   = model_fwd
turn_accum      = 0.0           # how much we’ve spun (radians)
prev_yaw        = p.getEulerFromQuaternion(
                     p.getBasePositionAndOrientation(env.robot)[1]
                  )[2]
# record where we started the turn
start_yaw       = prev_yaw     

print("Press ← or → to spin.  Forward is automatic.")

while True:
    # 1) poll keys
    keys = p.getKeyboardEvents()
    if p.B3G_LEFT_ARROW  in keys and mode == "forward":
        mode          = "left"
        current_model = model_left
        turn_accum    = 0.0
        # reset yaw trackers
        _, ori        = p.getBasePositionAndOrientation(env.robot)
        prev_yaw      = math.atan2(ori[2], ori[3]) * 2  # alt: p.getEulerFromQuaternion
        start_yaw     = prev_yaw

    if p.B3G_RIGHT_ARROW in keys and mode == "forward":
        mode          = "right"
        current_model = model_right
        turn_accum    = 0.0
        _, ori        = p.getBasePositionAndOrientation(env.robot)
        prev_yaw      = math.atan2(ori[2], ori[3]) * 2
        start_yaw     = prev_yaw

    # 2) choose command bit if you need it
    cmd = 0.0
    if mode == "left":  cmd = -1.0
    if mode == "right": cmd = +1.0

    # 3) integrate yaw change if we’re spinning
    if mode in ("left","right"):
        # get fresh yaw
        _, ori    = p.getBasePositionAndOrientation(env.robot)
        _,_,yaw   = p.getEulerFromQuaternion(ori)
        # compute smallest delta angle between prev_yaw→yaw
        delta     = math.atan2(math.sin(yaw-prev_yaw), math.cos(yaw-prev_yaw))
        turn_accum += delta
        prev_yaw   = yaw

        # if we’ve spun full circle, go forward
        if abs(turn_accum) >= 2*math.pi - 0.1:  
            mode          = "forward"
            current_model = model_fwd

    # 4) build the obs for this policy (28 vs 29 dims)
    want = current_model.observation_space.shape[0]
    if want == raw_obs.shape[0]:
        model_obs = raw_obs
    else:
        model_obs = np.concatenate(([cmd], raw_obs), axis=0)

    # 5) predict & step
    action, _      = current_model.predict(model_obs, deterministic=True)
    raw_obs, *__, info = env.step(action)
    env.render()

    # 6) handle end of episode
    done, truncated = False, False
    # your step already returns these; if needed, unpack them and reset here
    if done or truncated:
        raw_obs, _ = env.reset()
        mode       = "forward"
        current_model = model_fwd
        turn_accum    = 0.0
        _,ori         = p.getBasePositionAndOrientation(env.robot)
        prev_yaw      = p.getEulerFromQuaternion(ori)[2]
        start_yaw     = prev_yaw
