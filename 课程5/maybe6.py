import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import path_planning as pp

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


def get_goal_position(img):
    goal_disk_mask = cv2.inRange(img, np.array([0, 240, 0]), np.array([30, 255, 30]))
    contours, _  = cv2.findContours(goal_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
    front_disk_mask = cv2.inRange(img, np.array([238, 0, 0]), np.array([255, 30, 30]))
    contours, _  = cv2.findContours(front_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
    front_disk_mask = cv2.inRange(img, np.array([0, 0, 238]), np.array([30, 30, 255]))
   # plt.imshow(front_disk_mask)
    contours, _  = cv2.findContours(front_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
        return orientation2 - orientaion1
    else:
        return orientaion1 - orientation2







print('program started')

sim.simxFinish(-1)

clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to CoppeliaSim


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
    while counter > 0 and res == sim.simx_return_ok:
        res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_buffer)
        time.sleep(0.05)
        sim.simxSynchronousTrigger(clientID)
        counter = counter -1

    image = image_post_processing(image, resolution)
    robot_position = get_robot_center_position(image)
    img_obs = pp.obstacles_grid(image)
    plt.imshow(img_obs, cmap="gray")
    plt.show()
    goal_position = get_goal_position(image)
    t0 = time.time()
    return_code, path = pp.search(img_obs,  robot_position, goal_position)
    fig1 = plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.plot(path[:, 1], path[:, 0], '-r')
    plt.show()


    speed_controller = PIControl(0.4, 0.001, 5, 1)
    orient_controller = PIControl(1, 0.0, 5, 1)


    currentSimulationTime = 0
    tick = time.time()
    is_near_detected = False

    phase = 0

    point_counter = 0
    while currentSimulationTime < 600:
        
        res_img, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_buffer)
        currentSimulationTime = sim.simxGetLastCmdTime(clientID) * 0.001
        if res_img == sim.simx_return_ok:
            image = image_post_processing(image, resolution)
            robot_position = get_robot_center_position(image)
            robot_direction = get_robot_orientation(image)


            dist = get_distance(robot_position, goal_position)
            print("与目标距离："+str(dist))

            if abs(dist) < 20:
                break

            temp_goal = path[point_counter]

            dist0 = get_distance(robot_position, temp_goal)
            
            if abs(dist0) < 60 and point_counter < path.shape[0]-1:
                point_counter = point_counter + 1
                continue
            
            target_orient = get_orientation(robot_position, temp_goal)
            if point_counter == 40:
                sss=1
            print("路径点："+str(point_counter))
            print("机器人方向：" + str(robot_direction) + ", 目标方向：" + str(target_orient))


            v = speed_controller.control(dist0)
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

