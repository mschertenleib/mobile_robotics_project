import time
import numpy as np
import math
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import matplotlib.pyplot as plt
import os
import sys
import scipy
import random
from tdmclient import ClientAsync


# Function to set Thymio speed
def set_motor_speed(left_speed, right_speed):
    return {
        "motor.left.target": [left_speed],
        "motor.right.target": [right_speed],
    }


# Function to set Thymio to specific left and right speed
async def move_forward(node, client, motor_speedl, motor_speedr):
    # Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(motor_speedl, motor_speedr))

# Function to rotate Thymio to the right
async def turn_right(node, client, motor_speed):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(motor_speed, -motor_speed))

# Function to rotate Thymio to the left
async def turn_left(node, client, motor_speed):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(-motor_speed, motor_speed))

# Function to set Thymio back 
async def step_back(node, client, motor_speed):
    # Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(-motor_speed, -motor_speed))

# Function to detect whether the Thymio is in goal pursuit mode or obstacle avoidance mode and avoid obstacles. 
async def avoid_obstacles(node, client, num_samples_since_last_obstacle, side, obst, HTobst, LTobst, motor_speed,obst_gain) -> int:
    """# FIXME: comment
    
    """
    # Obstacle detection using proximals sensors
    # If an obstacle is detected, we switch to obstacle avoidance mode instead of goal tracking mode.
    if ((obst[0] > HTobst or obst[1] > HTobst) and (obst[0] > obst[2] or obst[1] > obst[2])):
        # Different cases are distinguished
        # Case 0 where the obstacle is on the left
        num_samples_since_last_obstacle = 0
        case = 0

    elif ((obst[4] > HTobst or obst[3] > HTobst) and (obst[4] > obst[2] or obst[3] > obst[2])):
        # Case 1 where the obstacle is on the right
        num_samples_since_last_obstacle = 0
        case = 1

    elif (obst[2] > HTobst and (obst[0] < obst[2] or obst[4] < obst[2]) and (
            obst[1] <= obst[2] or obst[1] <= obst[2])):
        # Case 2 where the obstacle is in front
        num_samples_since_last_obstacle = 0
        case = 2
    else:
        case = 3
        

    # Obstacle avoidance
    if case == 0 or case == 1:
        # Obstacle is to your left or right.
        # Obstacle avoidance: accelerate the wheel near the obstacle to turn in the opposite direction to the obstacle.
        if case == 0:
            # await turn_right(node, client, obst_gain * ((obst[0] + obst[1]) // 100))
            base_speed = obst_gain * ((obst[0] + obst[1]) // 100)
            await move_forward(node, client, base_speed, int(base_speed * 0.1))

        elif case == 1:
            # await turn_left(node, client, obst_gain * ((obst[4] + obst[3]) // 100))
            base_speed = obst_gain * ((obst[4] + obst[3]) // 100)
            await move_forward(node, client, int(base_speed * 0.1), base_speed)

    elif case == 2:
        # Obstacle is in front of the Thymio.
        # Avoid the obstacle in front by backing up and then turning randomly on one side.
        if side:
            await step_back(node, client, motor_speed)
            await turn_left(node, client, motor_speed)

        elif not side:
            await step_back(node, client, motor_speed)
            await turn_right(node, client, motor_speed)

    elif num_samples_since_last_obstacle >= 0:
        await move_forward(node, client, motor_speed, motor_speed)

    return num_samples_since_last_obstacle
