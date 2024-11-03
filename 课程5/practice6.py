import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

def is_near(robot_position, obstacle_bound, checkR):
    B1 = [robot_position[0] - checkR, robot_position[1] - checkR, robot_position[0] + checkR, robot_position[1] + checkR]
    B2 = obstacle_bound
    if B1[0] >= B2[2] or B1[2] <= B2[0] or B1[3] <= B2[1] or B1[1]>=B2[3]:
        return False
    else:
        return True


def image_post_processing(image, resolution):
    image = np.array(image, dtype=np.uint8)
    image.resize([resolution[1], resolution[0], 3])
    image = np.flipud(image)
    return  image

def get_obstacle_boundary(img):
    obstacle_disk_mask = cv2.inRange(img, np.array([90, 226, 151]), np.array([96, 232, 157]))
    im2, contours, hierarchy = cv2.findContours(obstacle_disk_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始值
    min_x = 255
    max_x = 0
    min_y = 255
    max_y = 0
    for c in contours:
        [rows, cols, numel] = c.shape
        for i in range(rows):
            for j in range(cols):
                #障碍物中的像素点x， y坐标
                p_x = c[i, j, 0]
                p_y = c[i, j, 1]
                #更新
                min_x = min(min_x, p_x)
                max_x = max(max_x, p_x)
                min_y = min(min_y, p_y)
                max_y = max(max_y, p_y)
    return min_x, min_y, max_x, max_y


def get_goal_position(img):
    goal_disk_mask = cv2.inRange(img, np.array([0, 240, 0]), np.array([30, 255, 30]))
    im2, contours, hierarchy = cv2.findContours(goal_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #no need for the for loop but it is used to avoid errors
    cY = 0
    cX = 0
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return np.array([cY,cX])


def get_front_disk_position (img):
    front_disk_mask = cv2.inRange(img, np.array([235, 0, 0]), np.array([255, 30, 30]))
    im2, contours, hierarchy = cv2.findContours(front_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #no need for the for loop but it is used to avoid errors
    cY = 0
    cX = 0
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return np.array([cY,cX])


def get_back_disk_position (img):
    front_disk_mask = cv2.inRange(img, np.array([0, 0, 235]), np.array([30, 30, 255]))
   # plt.imshow(front_disk_mask)
    im2, contours, hierarchy = cv2.findContours(front_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #no need for the for loop but it is used to avoid errors
    cY = 0
    cX = 0
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return np.array([cY,cX])


def get_robot_center_position(img):
    front_position = get_front_disk_position(img)
    back_position = get_back_disk_position(img)
    return  (front_position+back_position)/2


def get_distance(point0, point1):
    return math.sqrt((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2)

def get_orientation(point0, point1):
    orientation = np.arctan2(point0[1] - point1[1],
                             point0[0] - point1[0])
    return orientation


def get_robot_orientation(img):
    front_position = get_front_disk_position(img)
    back_position = get_back_disk_position(img)
    orientation = get_orientation(back_position, front_position)
    return orientation


def ponieer_robot(v_target, omg_target):
    wheel_distance = 0.2
    wheel_radius = 0.04
    v_left = v_target + omg_target * wheel_distance
    v_right = v_target - omg_target * wheel_distance
    theta_left = v_left / wheel_radius
    theta_right = v_right / wheel_radius


    return theta_left, theta_right


def get_turn_direction(orientaion1, orientation2):
    # direction > 0 means clock-wise, < 0  means counter-clock wise
    if abs(orientaion1 - orientation2) > math.pi:
        angle = 2*math.pi - abs(orientation2 - orientaion1)
        if orientation2 - orientaion1 > 0:
            return angle
        else:
            return -angle
    else:
        return orientaion1 - orientation2







print('program started')

sim.simxFinish(-1)

clientID = sim.simxStart('127.0.0.1', -3, True, True, 5000, 5) # Connect to CoppeliaSim


class PIControl():
    def __init__(self, kp, ki, n, max_signal):
        self.kp = kp
        self.ki = ki
        self.error = []
        self.n = n
        self.counter = 0
        self.max_signal = max_signal
        for i in range(n):
            self.error.append(0)

    def control(self, error):
        self.error[self.counter] = error
        self.counter = self.counter + 1
        if self.counter >= self.n:
            self.counter = 0

        sum_error = 0
        for i in range(self.n):
            sum_error = sum_error + self.error[i]

        control_signal = self.kp*error + self.ki*sum_error
        if control_signal > self.max_signal:
            control_signal = self.max_signal
        if control_signal < -self.max_signal:
            control_signal = -self.max_signal

        return control_signal




if clientID != -1:
    print('Connected to remote API server')
    # file_handle = open('report.txt', 'w')
    sim.simxSynchronous(clientID, True)
    # start the simulation:
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    res, camera0_handle = sim.simxGetObjectHandle(clientID, 'top_view_camera', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok:
        print("Failed to get top_view_camera handle")
    res, bubbleRobBase = sim.simxGetObjectHandle(clientID, 'bubbleRob', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")
    res, leftMotor = sim.simxGetObjectHandle(clientID, 'bubbleRob_leftMotor', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")
    res, rightMotor = sim.simxGetObjectHandle(clientID, 'bubbleRob_rightMotor', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")

    minMaxSpeed = [50*math.pi/180, 300*math.pi/180]
    speed = (minMaxSpeed[0] + minMaxSpeed[1]) * 0.6
    sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)

    err, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_streaming)
    sim.simxSynchronousTrigger(clientID)
    counter = 5
    while counter > 0:
        res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_buffer)
        time.sleep(0.05)
        sim.simxSynchronousTrigger(clientID)
        counter = counter -1

    image = image_post_processing(image, resolution)

    goal_position = get_goal_position(image)
    speed_controller = PIControl(0.6, 0.005, 5, 1)
    orient_controller = PIControl(1.5, 0.0, 5, 1)


    currentSimulationTime = 0
    tick = time.time()
    is_near_detected = False

    phase = 0
    while currentSimulationTime < 45:
        # res, result, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(
        #     clientID, noseSensor, sim.simx_opmode_buffer)
        res_img, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_buffer)
        currentSimulationTime = sim.simxGetLastCmdTime(clientID) * 0.001
        if res_img == sim.simx_return_ok:
            image = image_post_processing(image, resolution)
            robot_position = get_robot_center_position(image)
            robot_direction = get_robot_orientation(image)


            dist = get_distance(robot_position, goal_position)

            if abs(dist) < 20:
                break

            target_orient = get_orientation(robot_position, goal_position)
            print("机器人方向：" + str(robot_direction) + ", 目标方向：" + str(target_orient))

            v = speed_controller.control(dist)
            omg = orient_controller.control(get_turn_direction(robot_direction, target_orient))

            theta_left, theta_right = ponieer_robot(v, omg)
            sim.simxSetJointTargetVelocity(clientID, leftMotor, theta_left, sim.simx_opmode_oneshot)
            sim.simxSetJointTargetVelocity(clientID, rightMotor, theta_right, sim.simx_opmode_oneshot)

        else:
            print("failed to get proximity sensor result")

        sim.simxSynchronousTrigger(clientID)
    print("耗时： " + str(time.time() - tick))

    res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_discontinue)
    sim.simxGetPingTime(clientID)
    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')

