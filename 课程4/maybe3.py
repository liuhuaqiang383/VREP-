import sim
import math
import numpy as np
import time
import cv2

def is_near(robot_position, obstacle_bound, check_radius):
    """Check if the robot is near the obstacle within a given radius."""
    robot_bound = [robot_position[0] - check_radius, robot_position[1] - check_radius, 
                   robot_position[0] + check_radius, robot_position[1] + check_radius]
    return not (robot_bound[0] >= obstacle_bound[2] or 
                robot_bound[2] <= obstacle_bound[0] or 
                robot_bound[3] <= obstacle_bound[1] or 
                robot_bound[1] >= obstacle_bound[3])

def image_post_processing(image, resolution):
    """Processes the raw image from the sensor."""
    image = np.array(image, dtype=np.uint8)
    image.resize([resolution[1], resolution[0], 3])
    image = np.flipud(image)
    
    return image

def get_obstacle_boundary(img):
    """Detects and returns the boundary of the obstacle."""
    obstacle_mask = cv2.inRange(img, np.array([90, 226, 151]), np.array([96, 232, 157]))
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, 0, 0]
    x, y, w, h = cv2.boundingRect(contours[0])
    return [x, y, x + w, y + h]

def get_disk_position(img, color_lower, color_upper):
    """Returns the center position of a colored disk based on color range."""
    mask = cv2.inRange(img, color_lower, color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return np.array([cX, cY])
    return np.array([0, 0])

def get_robot_center_position(img):
    """Calculates the robot's center position using front and back disk positions."""
    front_position = get_disk_position(img, np.array([240, 0, 0]), np.array([255, 30, 30]))
    back_position = get_disk_position(img, np.array([0, 0, 240]), np.array([30, 30, 255]))
    return (front_position + back_position) / 2

def get_robot_orientation(img):
    """Calculates the robot's orientation angle."""
    front_position = get_disk_position(img, np.array([240, 0, 0]), np.array([255, 30, 30]))
    back_position = get_disk_position(img, np.array([0, 0, 240]), np.array([30, 30, 255]))
    orientation = np.arctan2(back_position[1] - front_position[1], back_position[0] - front_position[0])
    return orientation * 180 / math.pi

print("Program started")

sim.simxFinish(-1)
clientID = sim.simxStart("127.0.0.1", 19997, True, True, 5000, 5)
if clientID != -1:
    print("Connected to remote API server")
    sim.simxSynchronous(clientID, True)
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    res, camera_handle = sim.simxGetObjectHandle(clientID, "top_view_camera", sim.simx_opmode_oneshot_wait)
    res, leftMotor = sim.simxGetObjectHandle(clientID, "BubbleRob_leftMotor", sim.simx_opmode_blocking)
    res, rightMotor = sim.simxGetObjectHandle(clientID, "BubbleRob_rightMotor", sim.simx_opmode_blocking)
    res, noseSensor = sim.simxGetObjectHandle(clientID, "BubbleRob_sensingNose", sim.simx_opmode_blocking)

    minMaxSpeed = [50 * math.pi / 180, 300 * math.pi / 180]
    #speed = (minMaxSpeed[0] + minMaxSpeed[1]) * 0.6
    speed = (minMaxSpeed[0] + minMaxSpeed[1]) * 1.2
    sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)

    # Initialize camera stream
    res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_streaming)
    sim.simxSynchronousTrigger(clientID)

    for _ in range(5):
        res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_buffer)
        time.sleep(0.05)
        sim.simxSynchronousTrigger(clientID)

    image = image_post_processing(image, resolution)
    obstacle_bound = get_obstacle_boundary(image)

    phase = 0
    currentSimulationTime = 0
    while currentSimulationTime < 45:
        res_img, resolution, image = sim.simxGetVisionSensorImage(clientID, camera_handle, 0, sim.simx_opmode_buffer)
        currentSimulationTime = sim.simxGetLastCmdTime(clientID) * 0.001
        if res_img == sim.simx_return_ok:
            image = image_post_processing(image, resolution)
            robot_position = get_robot_center_position(image)
            robot_near_obstacle = is_near(robot_position, obstacle_bound, 30)

            if robot_near_obstacle and phase == 0:
                phase = 1

            if phase == 0:  # Move forward
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
            elif phase == 1:  # Turn right
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
                if abs(get_robot_orientation(image) - 90) < 1:
                    phase = 2
            elif phase == 2:  # Move forward until clear
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
                if not robot_near_obstacle:
                    phase = 3
            elif phase == 3:  # Turn left back to the original direction
                sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
                if abs(get_robot_orientation(image)) < 1:
                    phase = 0

        sim.simxSynchronousTrigger(clientID)

    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
    sim.simxFinish(clientID)
else:
    print("Failed to connect to remote API server")
print("Program ended")
