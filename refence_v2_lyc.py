import csv
import cv2
import os, sys
import numpy as np
from path import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def show(label, info):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(label[:, 0], label[:, 1], label[:, 2], zdir="z", c="#FF5511", marker="o", s=0.1)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.title(info)
    plt.show()


def show_2D(label, info):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(np.arange(0, len(label)), label[:, 0], c="#FF5511", marker="o", s=4)
    ax.set(xlabel="X", ylabel="Y")
    plt.title(info)
    # plt.plot(np.nprange,np.array(label))
    plt.show()


def load_data():
    position_deltas = []
    depths = sorted(Path((r"F:/Toky/Dataset/UnityCam/Recordings002/depth/")).files('*.exr'))
    imgs = sorted(Path((r"F:/Toky/Dataset/UnityCam/Recordings002/photo/")).files('*.png'))
    with open(r"F:/Toky/Dataset/UnityCam/Recordings002/position_rotation.csv", encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        position_deltas.append(line[0:].split(" "))
    return imgs, depths, position_deltas


def depth_read(depth_img_file_name):
    depth_img = cv2.imread(depth_img_file_name, 0)  # , dtype=cv2.CV_32F
    return depth_img


def img_read(img_file_name):
    img = cv2.imread(img_file_name, 0)  # 读取灰度图像
    return img


# 这个脚本可以将像素平面的点根据内参，外参转换为3维空间中的点云
if __name__ == '__main__':
    imgs, depths, position_deltas = load_data()
    depth_li, T_li, imgs_li = [], [], []  # 深度图, 齐次变换矩阵(外参), RGB图
    for i in range(13, 16):  # 导入 13, 14, 15 一共三组
        Rm = R.from_quat([position_deltas[i][3], position_deltas[i][4], position_deltas[i][5], position_deltas[i][6]])
        rotation_matrix = Rm.as_matrix()
        rvec = rotation_matrix  # 四元数 转 3*3 选择矩阵
        a = np.hstack((rvec, np.array([position_deltas[i][0], position_deltas[i][1], position_deltas[i][2]],
                                      dtype=np.float32).reshape(3, 1)))
        T_li.append(np.vstack((a, [0, 0, 0, 1])))
        depth_li.append(depth_read(depths[i]))
        imgs_li.append(img_read(imgs[i]))
    # print(depth_li[0].shape)
    # print(T_li[0])
    # print(imgs_li[0].shape)

    depth_scale = 1000  # 暂定
    cx = 396
    cy = 317
    fx = 389.93593
    fy = 309.77969
    instincts = np.asarray([389.93593, 0, 396, 0, 309.77969, 317, 0, 0, 1]).reshape(3, 3)  # 将内参写成一个矩阵

    point_3D_list_0, point_3D_list_1, point_3D_list_2 = [], [], []
    depth_0, img_0, T_0 = depth_li[0], imgs_li[0], T_li[0]  # 13
    depth_1, img_1, T_1 = depth_li[1], imgs_li[1], T_li[1]  # 14
    depth_2, img_2, T_2 = depth_li[2], imgs_li[2], T_li[2]  # 15

    # 测试1: depth_0 投影到三维再投影回 depth_0, 即 自己投影自己
    '''
    num = 0
    img_error_all, img_error_mean, depth_error_all, depth_error_mean = 0, 0, 0, 0
    for v in range(0, depth_0.shape[0]):  # <-- error: u,v 写反了
        for u in range(0, depth_0.shape[1]):
            d = np.max(depth_0[v, u])  # --------- 三通道有点奇怪，不应该这样改
            if d == 0:
                continue  # 为0表示没有测量到
            point = []
            zc = float(d)
            point.append((u - cx) * zc / fx)  # 像素 -> x 相机
            point.append((v - cy) * zc / fy)  # 像素 -> y 相机
            point.append(zc)  # 像素 -> z 相机
            point.append(1)
            # 相机坐标系 -> 世界坐标系
            try:
                T_0_inv = np.linalg.inv(T_0)
            except:
                T_0_inv = np.linalg.pinv(T_0)
            point_3D = T_0_inv @ np.array(point).reshape(4, 1)  # <-- error: 外参需求逆
            point_3D_list_0.append(point_3D)
            
            # 反向投影
            point_camera = T_0 @ point_3D  # 世界 -> 相机
            point_camera = point_camera[0:3]
            pixel_coord = (1 / point_camera[2]) * instincts @ point_camera  # 相机 -> 像素
            u_k = int(np.round(pixel_coord[0]))  # 像素 x
            v_k = int(np.round(pixel_coord[1]))  # 像素 y
            
            # 投影失败的不计算
            if u_k >= depth_0.shape[1] or u_k < 0 or v_k >= depth_0.shape[0] or v_k < 0:
                continue
            
            # 计算误差
            img_error_all += np.abs(float(img_0[v_k, u_k]) - float(img_0[v, u]))
            depth_error_all += np.abs(zc - float(np.max(depth_0[v_k, u_k])))
            num += 1
    img_error_mean = img_error_all / num
    depth_error_mean = depth_error_all / num
    print('success map num: ', num)
    print('img error all: ', img_error_all, ', img error mean: ', img_error_mean)
    print('depth error all: ', depth_error_all, ', depth error mean: ', depth_error_mean)
    '''

    # 测试2: depth_0 投影到世界坐标系, 然后移动, 再投影到 depth_1

    num = 0
    img_error_all, img_error_mean, depth_error_all, depth_error_mean = 0, 0, 0, 0

    # 计算在世界坐标系上的移动量
    move_temp = T_1 - T_0  # 平移
    move = np.array([move_temp[0, -1], move_temp[1, -1], move_temp[2, -1], 1]).reshape(4, 1)

    depth_error_list = []
    for v in range(0, depth_0.shape[0], 2):
        for u in range(0, depth_0.shape[1], 2):
            d = np.max(depth_0[v, u])  # --------- 三通道有点奇怪，不应该这样改
            if d == 0:
                continue  # 为0表示没有测量到
            point = []
            zc = float(d)
            point.append((u - cx) * zc / fx)  # 像素 -> x 相机
            point.append((v - cy) * zc / fy)  # 像素 -> y 相机
            point.append(zc)  # 像素 -> z 相机
            point.append(1)
            # 相机坐标系 -> 世界坐标系
            try:
                T_0_inv = np.linalg.inv(T_0)
            except:
                T_0_inv = np.linalg.pinv(T_0)
            point_3D = T_0_inv @ np.array(point).reshape(4, 1)  # <-- error: 外参需求逆
            point_3D_list_0.append(point_3D)

            # 在世界坐标系上移动
            # point_3D_new = point_3D + move
            point_3D_new = point_3D

            # 反向投影
            try:
                T_1_inv = np.linalg.inv(T_1)
            except:
                T_1_inv = np.linalg.pinv(T_1)
            point_camera = T_1 @ point_3D_new  # 世界 -> 相机
            point_camera = point_camera[0:3]
            pixel_coord = (1 / point_camera[2]) * instincts @ point_camera  # 相机 -> 像素
            u_k = int(np.round(pixel_coord[0]))  # 像素 x
            v_k = int(np.round(pixel_coord[1]))  # 像素 y

            # 投影失败的不计算
            if u_k >= depth_1.shape[1] or u_k < 0 or v_k >= depth_1.shape[0] or v_k < 0:
                continue
            if float(img_1[v_k, u_k]) <= 5 or float(img_0[v, u]) <= 5:
                continue
            if float(depth_1[v_k, u_k]) <= 20 or float(depth_0[v, u]) <= 20:
                continue

            # 计算误差
            img_error_all += np.abs(float(img_1[v_k, u_k]) - float(img_0[v, u]))
            depth_error = np.abs(zc - float(np.max(depth_1[v_k, u_k])))
            depth_error_all += depth_error
            depth_error_list.append(depth_error)
            num += 1
    img_error_mean = img_error_all / num
    depth_error_mean = depth_error_all / num
    print('success map num: ', num)
    print('img error all: ', img_error_all, ', img error mean: ', img_error_mean)
    print('depth error all: ', depth_error_all, ', depth error mean: ', depth_error_mean)
    # a=np.array(depth_error_list)
    show_2D(np.array(depth_error_list).reshape(len(depth_error_list), 1), "depth_error")

    # 测试3: 直接基于 depth_0 的相机坐标系
    # 待写

    '''
    point_3D_list = []
    XYZ1c = []
    colors = []
    # for depth, img, T in zip(depth_li, imgs_li, T_li):
    depth_0, img_0, T_0 = depth_li[0], imgs_li[0], T_li[0]
    point_3D_list_1 = []

    depth_1, img_1, T_1 = depth_li[1], imgs_li[1], T_li[1]  # 需要提前知道下一个变换的位姿
    depth_2, img_2, T_2 = depth_li[2], imgs_li[2], T_li[2]
    point_3D_list_2 = []
    depth_error = 0
    img_error = 0
    num_projected_point = 0
    

            # 反向投影
            point_camera = np.linalg.inv(T_1) @ point_3D  # 反投影时使用
            # point_camera = T_1@ point_3D
            point_camera = point_camera[0:3]
            
            if v_k != v or u_k != u:
                print("v_k!=v")
            
            # depth_0 需要投影到  投影失败的则忽略
            # error: depth_0 -> depth_1, 因为是投影到 depth_1 上; u,v 上面修改了所以这里也要修改
            if u_k >= depth_1.shape[1] or u_k < 0 or v_k >= depth_1.shape[0] or v_k < 0:
                continue
            
            # 这个投影下来的坐标和谁进行比较呢？而且应该要保证投影的坐标也是在像素平面内部的，不然就视作没有成功投影的像素点
            # 现在有两个像素坐标，就可以计算灰度值的差了
            # 还是说需要记录下这些新的uk vk 的值，作为成功投影的像素点，再取一个有效性mask ，去计算经过mask 屏蔽过后的两张图像之间的整体相似性作为loss呢
            print(point_camera[2] - np.max(depth_1[u, v]))
            num_projected_point += 1
            depth_error = depth_error + np.abs(point_camera[2] - int(np.max(depth_1[u, v])))
            img_error = img_error + np.abs(int(img_0[u, v]) - int(img_1[u_k, v_k]))


    print(T_0, T_1)
    print(f'成功投影的点数：{num_projected_point}，深度误差：{depth_error}，图像灰度值误差：{img_error}，'
          f'总的像素点数: {depth_0.shape[0] * depth_0.shape[1]}')
    '''
