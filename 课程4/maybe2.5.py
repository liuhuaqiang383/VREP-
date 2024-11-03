import sim
import math
import time
import numpy as np
import cv2

# Function to preprocess the camera image
def image_post_processing(image, resolution):
    image = np.array(image, dtype=np.uint8)
    image.resize([resolution[1], resolution[0], 3])
    image = np.flipud(image)
    return image

# Function to detect obstacle boundaries from image
def get_obstacle_boundary(img):
    obstacle_mask = cv2.inRange(img, np.array([90, 226, 151]), np.array([96, 232, 157]))
    contours, _ = cv2.findContours(obstacle_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_x, min_y, max_x, max_y = 255, 255, 0, 0
    for c in contours:
        for point in c:
            p_x, p_y = point[0]
            min_x, max_x = min(min_x, p_x), max(max_x, p_x)
            min_y, max_y = min(min_y, p_y), max(max_y, p_y)
    return [min_x, min_y, max_x, max_y]

print('Program started')
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print('Connected to remote API server')
    sim.simxSynchronous(clientID, True)

    # Handle Initialization
    err, camera_handle = sim.simxGetObjectHandle(clientID, 'top_view_camera', sim.simx_opmode_blocking)
    err, leftMotor = sim.simxGetObjectHandle(clientID, 'BubbleRob_leftMotor', sim.simx_opmode_blocking)
    err, rightMotor = sim.simxGetObjectHandle(clientID, 'BubbleRob_rightMotor', sim.simx_opmode_blocking)

    # Start the simulation
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    # Camera and motors setup for streaming
    sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)

    # Timing Variables
    currentSimulationTime = 0
    detected_wall_time = backward_start_time = -1
    speed = 175 / 180 * math.pi

    while currentSimulationTime < 45:
        res_img, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_buffer)
        currentSimulationTime = sim.simxGetLastCmdTime(clientID) * 0.001

        if res_img == sim.simx_return_ok:
            image = image_post_processing(image, resolution)
            obstacle_bound = get_obstacle_boundary(image)

            # Control logic for obstacle avoidance
            if detected_wall_time < 0 and obstacle_bound:
                detected_wall_time = currentSimulationTime
                backward_start_time = detected_wall_time + 0.5
                turning_start_time = backward_start_time + 1.5
                straight_start_time = turning_start_time + 0.5
                reset_direction_time = straight_start_time + 1.5

            # Control stages
            if currentSimulationTime < backward_start_time:
                # Move forward
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
            elif currentSimulationTime < turning_start_time:
                # Move backward
                sim.simxSetJointTargetVelocity(clientID, leftMotor, -speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, -speed, sim.simx_opmode_oneshot)
            elif currentSimulationTime < straight_start_time:
                # Turn
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
            elif currentSimulationTime < reset_direction_time:
                # Go straight after turning
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
            else:
                # Reset to original direction if needed
                sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)

        sim.simxSynchronousTrigger(clientID)

    # Stop simulation and disconnect
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')
