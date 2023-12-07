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


# Function to set Thymio colors leds
def set_leds(ledsbl, ledsbr, ledst):
    return {
        "leds.bottom.left": ledsbl,
        "leds.bottom.right": ledsbr,
        # "leds.circle":ledsc,
        "leds.top": ledst,
    }


# Function to set Thymio back for a certain period of time
async def move_forward(node, client, motor_speedl, motor_speedr):
    # Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(motor_speedl, motor_speedr))
    # Allow time for the forward movement
    await client.sleep(0.1)
    # Stop the motors await.node.set_variables(set_motor_speed(0, 0))  


# Function to rotate Thymio by 90 degrees clockwise/to turn left
async def rotate_90_degrees_clockwise(node, client, motor_speed, rotation_time):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(motor_speed * 2, -motor_speed * 2))
    # Allow time for the rotation 
    await client.sleep(rotation_time)
    # Stop the motors 
    await node.set_variables(set_motor_speed(0, 0))


# Function to rotate Thymio by 90 degrees counterclockwise/to turn right
async def rotate_90_degrees_counterclockwise(node, client, motor_speed, rotation_time):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(-motor_speed * 2, motor_speed * 2))
    # Allow time for the rotation 
    await client.sleep(rotation_time)
    # Stop the motors
    await node.set_variables(set_motor_speed(0, 0))


# Function to set Thymio back for a certain period of time
async def step_back(node, client, motor_speed, step_back_time):
    # Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(-motor_speed, -motor_speed))
    # Allow time for the step back
    await client.sleep(step_back_time)
    # Stop the motors
    await node.set_variables(set_motor_speed(0, 0))


# Function to detect if the Thymio is in the goal tracking mode or the obstacle avoidance mode
async def avoid_obstacles(node, client, state, side, obst, HTobst, LTobst, motor_speed, step_back_time, obst_gain,
                          rotation_time):
    # Obstacle detection using proximals sensors
    if state == 0:
        # If an obstacle is detected, we switch to obstacle avoidance mode instead of goal tracking mode.
        if ((obst[0] > HTobst or obst[1] > HTobst) and (obst[0] > obst[2] or obst[1] > obst[2])):
            # Different cases are distinguished
            # Case 0 where the obstacle is on the left
            case = 0
            state = 1

        elif ((obst[4] > HTobst or obst[3] > HTobst) and (obst[4] > obst[2] or obst[3] > obst[2])):
            # Case 1 where the obstacle is on the right
            case = 1
            state = 1

        elif (obst[2] > HTobst and (obst[0] < obst[2] or obst[4] < obst[2])):
            # Case 2 where the obstacle is in front
            case = 2
            state = 1

    elif state == 1:
        if obst[0] < LTobst:
            if obst[4] < LTobst:
                if obst[2] < LTobst:
                    # If obstacle is not detected, we switch from obstacle avoidance mode to goal tracking mode.
                    state = 0

    # Obstacle avoidance   
    if state == 0:
        # goal tracking: turn toward the goal
        # await move_forward(node, client, motor_speed, motor_speed)
        # await node.set_variables(set_motor_speed(0, 0))
        pass

    else:
        if case == 0 or case == 1:
            # Obstacle is to your left or right.
            # Obstacle avoidance: accelerate the wheel near the obstacle to turn in the opposite direction to the obstacle.
            await move_forward(node, client,
                               motor_speed + obst_gain * ((obst[0] + obst[1]) // 100),
                               motor_speed + obst_gain * ((obst[4] + obst[3]) // 100))

        elif case == 2:
            # Obstacle is in front of the Thymio.
            # Avoid the obstacle in front by backing up and then turning randomly on one side. 
            if side:
                await step_back(node, client, motor_speed, step_back_time)
                await rotate_90_degrees_counterclockwise(node, client, motor_speed, rotation_time)

            elif not side:
                await step_back(node, client, motor_speed, step_back_time)
                await rotate_90_degrees_clockwise(node, client, motor_speed, rotation_time)
