import numpy as np
from queue import PriorityQueue
import cv2


#transform from image frame to vrep frame
def transform_points_from_image2real (points):
    if points.ndim < 2:
        flipped = np.flipud(points)
    else:
        flipped = np.fliplr(points)
    scale = 5/445
    points2send = (flipped*-scale) + np.array([2.0555+0.75280899, -2.0500+4.96629213])
    return points2send



def smooth(path, grid, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001, number_of_iter=1e3):
    newpath = np.copy(path).astype('float64')

    def get_near_obstacles(point, area=5):
        x_start = int(max(point[0] - area, 0))
        x_end = int(point[0] + area)
        y_start = int(max(point[1] - area, 0))
        y_end = int(point[1] + area)
        points = np.argwhere(grid[x_start:x_end, y_start:y_end] < 128)
        points[:, 0] += x_start
        points[:, 1] += y_start
        if not points.size:
            points = point.copy()
        return points

    def near_obstacles(point, half_kernel=2):
        x_start = int(max(point[0] - half_kernel, 0))
        x_end = int(point[0] + half_kernel)
        y_start = int(max(point[1] - half_kernel, 0))
        y_end = int(point[1] + half_kernel)
        return np.any(grid[x_start:x_end, y_start:y_end] < 128)

    error = np.ones(path.shape[0]) * tolerance + tolerance
    num_points = path.shape[0]
    for count in range(int(number_of_iter)):
        for i in range(1, num_points - 1):
            old_val = np.copy(newpath[i])
            update1 = weight_data * (path[i] - newpath[i])
            update2 = weight_smooth * (newpath[i - 1] + newpath[i + 1] - 2 * newpath[i])
            newpath[i] += update1 + update2
            if near_obstacles(newpath[i], half_kernel=35):
                newpath[i] = old_val
            error[i] = np.abs(np.mean(old_val - newpath[i]))
        if np.mean(error) < tolerance:
            break
    print(count)
    return newpath


# 输入参数：
# grid: 障碍物的掩码图，障碍物应显示为黑色，其余均显示白色
# init: 初始位置，np.array([H, W]), H为初始点像素在高度方向的值，W为初始点像素在宽度方向的值
# goal：目标位置，np.array([H, W]), H为目标点像素在高度方向的值，W为目标点像素在宽度方向的值
# cost，D：精度控制参数，可使用默认值
# fnc：规划采用的核函数
# 输出参数：
# returnCode: 0表示成功，1表示规划失败
# path: N*2 尺寸的 np.array， 即表示路径上的N个像素点，[ [H1, W1],  [H2, W2], ... [Hn, Wn] ]
def search(grid, init, goal, cost = 1, D = 1, fnc='Manhattan'):
    D2 = 1
    return_code = 1
    init = tuple(init.astype('int'))
    goal = tuple(goal.astype('int'))
    def Euclidean_fnc(current_indx, goal_indx ,D = 1):
        return np.sqrt( ( (current_indx[0 ] -goal_indx[0] )**2 + (current_indx[1 ] -goal_indx[1] )**2 ) )
    def Manhattan_fnc(current_indx, goal_indx ,D = 1):
        dx = np.sqrt((current_indx[0 ] -goal_indx[0] )**2)
        dy = np.sqrt((current_indx[1 ] -goal_indx[1] )**2)
        return D * (dx + dy)
    def Diagonal_fnc(current_indx, goal_indx ,D = 1):
        dx = np.sqrt((current_indx[0 ] -goal_indx[0] )**2)
        dy = np.sqrt((current_indx[1 ] -goal_indx[1] )**2)
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    if fnc == 'Euclidean':
        hueristic_fnc = Euclidean_fnc
    elif fnc == "Manhattan":
        hueristic_fnc = Manhattan_fnc
    elif fnc == "Diagonal" :
        hueristic_fnc = Diagonal_fnc


    def near_obstacles(point, half_kernel = 5):
        x_start = int(max(point[0] - half_kernel, 0))
        x_end = int(min(point[0] + half_kernel, grid.shape[0]))
        y_start = int(max(point[1] - half_kernel, 0))
        y_end = int(min(point[1] + half_kernel, grid.shape[1]))
        return np.any(grid[x_start:x_end, y_start:y_end ] <128)

    def delta_gain(gain = 1):
        delta = np.array([[-1, 0], # go up
                          [-1 ,-1], # up left
                          [ 0 ,-1], # go left
                          [ 1 ,-1], # down left
                          [ 1, 0], # go down
                          [ 1, 1],  # down right
                          [ 0, 1], # go right
                          [-1, 1] # up right
                          ])
        return delta*gain

    delta = delta_gain(gain = 5)
    front = PriorityQueue()
    G = 0
    H = hueristic_fnc(init, goal, D)
    F = G+ H
    front.put((F, G, init))
    discovered = []
    discovered.append(init)

    actions = np.ones_like(grid) * -1
    count = 0
    path = []

    def policy_draw(indx):
        indx_old = tuple(indx)
        indx_new = tuple(indx)
        path.append(tuple(goal))
        while indx_new != init:
            indx_new = tuple(np.array(indx_old) - delta[int(actions[indx_old])])
            path.append(indx_new)
            indx_old = indx_new

    while not front.empty():
        front_element = front.get()
        G = front_element[1]
        indx = front_element[2]
        if ((indx[0] >= goal[0] - 20) and (indx[0] < goal[0] + 20)) and (
                (indx[1] >= goal[1] - 20) and (indx[1] < goal[1] + 20)):
            policy_draw(indx)
            print("found goal")
            print(count)
            print(front_element)
            return_code = 0
            break
        else:
            for y in range(len(delta)):
                indx_new = tuple(indx + delta[y])
                if ((np.any(np.array(indx_new) < 0)) or (indx_new[0] > grid.shape[0] - 1) or (
                        indx_new[1] > grid.shape[1] - 1)):
                    continue
                if (grid[indx_new] >= 128) and (indx_new not in discovered):
                    count += 1
                    # if the obstacle is inside the robot :D, have a really high cost
                    if near_obstacles(indx_new, half_kernel=35):
                        g_new = G + 1500 * cost
                    # if the obstacle is about a robot's length near it , have a high cost
                    elif near_obstacles(indx_new, half_kernel=70):
                        g_new = G + 15 * cost
                    # as before
                    elif near_obstacles(indx_new, half_kernel=100):
                        g_new = G + 10 * cost
                    # as before
                    elif near_obstacles(indx_new, half_kernel=110):
                        g_new = G + 5 * cost
                    else:
                        g_new = G + cost
                    # trying to increase the cost of rapidly changing direction
                    if y == actions[indx]:
                        g_new = g_new
                    elif (y - 1) % len(delta) == actions[indx] or (y + 1) % len(delta) == actions[indx]:
                        g_new = g_new + 5 * cost
                    else:
                        g_new = g_new + 10 * cost
                    h_new = hueristic_fnc(indx_new, goal, D)
                    f_new = (g_new + h_new) - 0.0001 * count
                    front.put((f_new, g_new, indx_new))
                    discovered.append(indx_new)
                    actions[indx_new] = y
    else:
        print(count)
        print("fail")
    path = np.array(path[::-1])
    newpath = path
    if return_code == 0:
        newpath = smooth(path, grid, weight_data=0.1, weight_smooth=0.65, number_of_iter=1000)
    return return_code, newpath


# 获取障碍物掩码图
def obstacles_grid(img):
    # getting the walls
    mask_wall = cv2.inRange(img, np.array([230, 230, 230]),np.array([240, 240, 240]))
    # getting the rims
    mask_rim = cv2.inRange(img, 0, 0)
    mask_total = cv2.bitwise_or(mask_wall,mask_rim,mask_rim)
    mask_total = cv2.bitwise_not(mask_total)
    return mask_total
