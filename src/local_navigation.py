import numpy as np


# Function to set Thymio speed
def set_motor_speed(left_speed, right_speed):
    return {
        "motor.left.target": [left_speed],
        "motor.right.target": [right_speed],
    }


# Function to set Thymio to specific left and right speed
async def move_forward(node, motor_speedl, motor_speedr):
    # Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(motor_speedl, motor_speedr))


# Function to rotate Thymio to the right
async def turn_right(node, motor_speed):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(motor_speed, -motor_speed))


# Function to rotate Thymio to the left
async def turn_left(node, motor_speed):
    # Set the motor speeds in opposite directions to induce rotation
    await node.set_variables(set_motor_speed(-motor_speed, motor_speed))


# Function to set Thymio back
async def step_back(node, motor_speed):
    # Set the motor speeds in opposite signs to induce step back
    await node.set_variables(set_motor_speed(-motor_speed, -motor_speed))


async def avoid_obstacles(node, num_samples_since_last_obstacle: int, prox: list[int]) -> int:
    """
    Detects whether the Thymio is in goal pursuit mode or obstacle avoidance mode, and avoid obstacles.
    Returns the new value of num_samples_since_last_obstacle, which will be set to 0 if an obstacle was detected, else
    left unchanged.
    """

    motor_speed = 200  # Reference speed of the robot for avoidance
    threshold = 17  # Sensor threshold for obstacle detection
    prox_gain = 0.07

    # Obstacle detection using proximals sensors
    # If an obstacle is detected, we switch to obstacle avoidance mode instead of goal tracking mode.
    if (prox[0] > threshold or prox[1] > threshold) and (prox[0] > prox[2] or prox[1] > prox[2]):
        # Different cases are distinguished
        # Case 0 where the obstacle is on the left
        num_samples_since_last_obstacle = 0
        case = 0

    elif (prox[4] > threshold or prox[3] > threshold) and (prox[4] > prox[2] or prox[3] > prox[2]):
        # Case 1 where the obstacle is on the right
        num_samples_since_last_obstacle = 0
        case = 1

    elif (prox[2] > threshold and (prox[0] < prox[2] or prox[4] < prox[2]) and (
            prox[1] <= prox[2] or prox[1] <= prox[2])):
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
            base_speed = prox_gain * (prox[0] + prox[1])
            await move_forward(node, int(base_speed), int(base_speed * 0.1))

        elif case == 1:
            base_speed = prox_gain * (prox[4] + prox[3])
            await move_forward(node, int(base_speed * 0.1), int(base_speed))

    elif case == 2:
        # Obstacle is in front of the Thymio.
        # Avoid the obstacle in front by backing up and then turning randomly on one side.
        if np.random.randint(2) == 0:
            await step_back(node, motor_speed)
            await turn_left(node, motor_speed)
        else:
            await step_back(node, motor_speed)
            await turn_right(node, motor_speed)

    elif num_samples_since_last_obstacle >= 0:
        await move_forward(node, motor_speed, motor_speed)

    return num_samples_since_last_obstacle
