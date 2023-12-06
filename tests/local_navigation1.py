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
    "motor.left.target" : [left_speed],
    "motor.right.target" : [right_speed],
    }

# Function to set Thymio colors leds
def set_leds(ledsbl, ledsbr, ledst):
    return {
    "leds.bottom.left": ledsbl,       
    "leds.bottom.right":ledsbr,
    #"leds.circle":ledsc,
    "leds.top":ledst,
    }
    
# Function to set Thymio back 
async def move_forward(node,client,motor_speedl, motor_speedr):
    #Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(motor_speedl, motor_speedr))  
    
# Function to rotate Thymio to the right
async def turn_right(node,client,motor_speed):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(motor_speed, -motor_speed))

# Function to rotate Thymio to the left
async def turn_left(node,client,motor_speed):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(-motor_speed, motor_speed))  

# Function to set Thymio back for a certain period of time
async def step_back(node,client,motor_speed):
    #Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(-motor_speed, -motor_speed))   
    
# Function to detect whether the Thymio is in goal pursuit mode or obstacle avoidance mode and avoid obstacles. 
async def avoid_obstacles(node,client,state,side,obst, HTobst, LTobst, motor_speed, obst_gain):     
    # Obstacle detection using proximals sensors
    if state == 0:
    #If an obstacle is detected, we switch to obstacle avoidance mode instead of goal tracking mode. 
        if ((obst[0] > HTobst or obst[1]> HTobst) and (obst[0]> obst[2] or obst[1]>obst[2])):
            #Different cases are distinguished 
            #Case 0 where the obstacle is on the left  
            case = 0 
            state = 1
            
        elif ((obst[4] > HTobst or obst[3]> HTobst )and (obst[4]> obst[2] or obst[3] > obst[2])):
            #Case 1 where the obstacle is on the right  
            case = 1
            state = 1
            
        elif (obst[2]> HTobst and (obst[0]< obst[2] or obst[4]< obst[2]) and (obst[1]<= obst[2] or obst[1]<= obst[2])): 
            #Case 2 where the obstacle is in front  
            case = 2 
            state = 1
            
    elif state == 1:
        if obst[0] < LTobst :
            if obst[4] < LTobst :
                if obst[2] < LTobst :
                    #If obstacle is not detected, we switch from obstacle avoidance mode to goal tracking mode. 
                    state = 0
    
    # Obstacle avoidance   
    if state == 0:
        # goal tracking: turn toward the goal
        await move_forward(node,client,motor_speed, motor_speed)
        
    else:
        if case == 0 or case == 1:
            # Obstacle is to your left or right.
            # Obstacle avoidance: accelerate the wheel near the obstacle to turn in the opposite direction to the obstacle.
            if case == 0:
                await turn_right(node,client,obst_gain * ((obst[0]+obst[1]) // 100))

            elif case == 1:     
                await turn_left(node,client,obst_gain * ((obst[4]+obst[3]) // 100))
          
            
        elif case == 2:
            # Obstacle is in front of the Thymio.
            # Avoid the obstacle in front by backing up and then turning randomly on one side. 
            if side:
                await step_back(node,client,motor_speed)
                await turn_left(node,client,motor_speed)
                
            elif not side:
                await step_back(node,client,motor_speed)
                await turn_right(node,client,motor_speed)