import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_front_disk_position (img):
    front_disk_mask = cv2.inRange(img, np.array([200, 0, 0]), np.array([255, 50, 50]))
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
    speed = (minMaxSpeed[0] + minMaxSpeed[1]) * 0.3

    res, result, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(
        clientID, noseSensor, sim.simx_opmode_streaming)
    err, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_streaming)
    # while not  res == sim.simx_return_ok:
    #     res, result, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(
    #         clientID, noseSensor, sim.simx_opmode_streaming)
    #     time.sleep(0.05)
    #     sim.simxSynchronousTrigger(clientID)

    currentSimulationTime = 0
    tick = time.time()
    while currentSimulationTime < 45:
        res, result, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = sim.simxReadProximitySensor(
            clientID, noseSensor, sim.simx_opmode_buffer)
        res_img, resolution, image = sim.simxGetVisionSensorImage(clientID, camera0_handle, 0, sim.simx_opmode_buffer)
        if res_img == sim.simx_return_ok:
            image = np.array(image, dtype=np.uint8)
            image.resize([resolution[1], resolution[0], 3])
            print(get_front_disk_position(image))
            plt.cla()
            plt.imshow(image)
            plt.pause(0.01)
        currentSimulationTime = sim.simxGetLastCmdTime(clientID) * 0.001


        if res == sim.simx_return_ok:

            if result :
                turnR = 0.2 + 0.04
                turnLength = turnR * 2 * math.pi / 4
                wheel_v_speed = speed * 0.08
                turn90degreeTime = turnLength / wheel_v_speed * 1.2
                avoidTime = 0.5 / wheel_v_speed * 1.5

                turnUntilTime1 = currentSimulationTime + turn90degreeTime
                turnUntilTime2 = turnUntilTime1 + avoidTime
                turnUntilTime3 = turnUntilTime2 + turn90degreeTime * 1.5
                print(turnUntilTime1)
                print(turnUntilTime2)
                print(turnUntilTime3)

            if turnUntilTime3 < currentSimulationTime:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
            elif currentSimulationTime <= turnUntilTime1:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_oneshot)
            elif currentSimulationTime <= turnUntilTime2:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, speed, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
            elif currentSimulationTime <= turnUntilTime3:
                sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, rightMotor, speed, sim.simx_opmode_oneshot)
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

