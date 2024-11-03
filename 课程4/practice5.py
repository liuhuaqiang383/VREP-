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
    min_x = 512
    max_x = 0
    min_y = 512
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

def get_front_disk_position (img):
    front_disk_mask = cv2.inRange(img, np.array([240, 0, 0]), np.array([255, 30, 30]))
    im2, contours, hierarchy = cv2.findContours(front_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #no need for the for loop but it is used to avoid errors
    cY = 0
    cX = 0


    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
    return np.array([cY,cX])


def get_back_disk_position (img):
    front_disk_mask = cv2.inRange(img, np.array([0, 0, 240]), np.array([30, 30, 255]))
   # plt.imshow(front_disk_mask)
    im2, contours, hierarchy = cv2.findContours(front_disk_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #no need for the for loop but it is used to avoid errors
    cY = 0
    cX = 0
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
    return np.array([cY,cX])


def get_robot_center_position(img):
    front_position = get_front_disk_position(img)
    back_position = get_back_disk_position(img)
    return  (front_position+back_position)/2

def get_robot_orientation(img):
    front_position = get_front_disk_position(img)
    back_position = get_back_disk_position(img)
    orientation = np.arctan2( back_position[1] - front_position[1],
                              back_position[0] - front_position[0])
    return orientation*180/math.pi



print('program started')

sim.simxFinish(-1)

clientID = sim.simxStart('127.0.0.1', -3, True, True, 5000, 5) # Connect to CoppeliaSim

if clientID != -1:
    print('Connected to remote API server')
    sim.simxSynchronous(clientID, True)
    # start the simulation:
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    res, camera0_handle = sim.simxGetObjectHandle(clientID, 'top_view_camera', sim.simx_opmode_oneshot_wait)
    if res != sim.simx_return_ok:
        print("Failed to get top_view_camera handle")
    res, bubbleRobBase = sim.simxGetObjectHandle(clientID, 'BubbleRob', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")
    res, leftMotor = sim.simxGetObjectHandle(clientID, 'BubbleRob_leftMotor', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")
    res, rightMotor = sim.simxGetObjectHandle(clientID, 'BubbleRob_rightMotor', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")
    res, noseSensor = sim.simxGetObjectHandle(clientID, 'BubbleRob_sensingNose', sim.simx_opmode_blocking)
    if res != sim.simx_return_ok:
        print("Failed to get object handle")
    minMaxSpeed = [50*math.pi/180, 300*math.pi/180]
    turnUntilTime1 = -1
    turnUntilTime2 = -1
    turnUntilTime3 = -1
    speed = (minMaxSpeed[0] + minMaxSpeed[1]) * 0.6
    sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)

    res, result, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(
        clientID, noseSensor, sim.simx_opmode_streaming)
    err, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_streaming)
    sim.simxSynchronousTrigger(clientID)
    counter = 5
    while counter > 0:
        res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_buffer)
        time.sleep(0.05)
        sim.simxSynchronousTrigger(clientID)
        counter = counter -1

    image = image_post_processing(image, resolution)

    bound_min_x, bound_min_y, bound_max_x, bound_max_y = get_obstacle_boundary(image)
    obstacle_bound = [bound_min_x, bound_min_y, bound_max_x, bound_max_y]


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
            print("位置：" + str(robot_position))

            result = is_near(robot_position, obstacle_bound, 30)
            if result and not is_near_detected:
                is_near_detected = True
                phase = 1

            if phase == 0:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
            elif phase == 1:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
                robot_direction = get_robot_orientation(image)
                if abs(robot_direction - 90) < 1:
                    phase = 2
            elif phase == 2:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
                if not result:
                    phase = 3
            elif phase == 3:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
                robot_direction = get_robot_orientation(image)
                print("方向：" + str(robot_direction))
                if abs(robot_direction) < 1:
                    phase = 0
        else:
            print("failed to get proximity sensor result")

        sim.simxSynchronousTrigger(clientID)
    print("耗时： " + str(time.time() - tick))
    res, result, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(
        clientID, noseSensor, sim.simx_opmode_discontinue)
    sim.simxGetPingTime(clientID)
    res, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_discontinue)

    sim.simxGetPingTime(clientID)
    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')

