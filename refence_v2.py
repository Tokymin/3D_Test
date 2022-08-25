import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from path import Path
from scipy.spatial.transform import Rotation as R

from refence_v2_lyc import show_2D


def show(label, info):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(label[:, 0], label[:, 1], label[:, 2], zdir="z", c="#FF5511", marker="o", s=0.1)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.title(info)
    plt.show()


def load_data():
    position_deltas = []
    # for name in os.walk():
    depths = sorted(Path((r"F:/Toky/Dataset/UnityCam/Recordings002/depth/")).files('*.exr'))
    imgs = sorted(Path((r"F:/Toky/Dataset/UnityCam/Recordings002/photo/")).files('*.png'))
    with open(r"F:/Toky/Dataset/UnityCam/Recordings002/position_rotation.csv", encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        position_deltas.append(line[0:].split(" "))
    return imgs, depths, position_deltas


def depth_read(depth_img_file_name):
    depth_img = cv2.imread(depth_img_file_name, -1)  # , dtype=cv2.CV_32F
    return depth_img * 100


def img_read(img_file_name):
    img = cv2.imread(img_file_name, 0)  # 直接读取灰度图像吧
    return img


# 这个脚本可以将像素平面的点根据内参，外参转换为3维空间中的点云
if __name__ == '__main__':
    imgs, depths, position_deltas = load_data()
    depth_li = []
    T_li = []
    imgs_li = []
    for i in range(13, 16):
        Rm = R.from_quat([position_deltas[i][3], position_deltas[i][4], position_deltas[i][5], position_deltas[i][6]])
        rotation_matrix = Rm.as_matrix()
        rvec = rotation_matrix  # 3*3,针对四元数的
        a = np.hstack((rvec, np.array([position_deltas[i][0], position_deltas[i][1], position_deltas[i][2]],
                                      dtype=np.float32).reshape(3, 1)))
        T_li.append(np.vstack(
            (a, [0, 0, 0, 1])))
        depth_li.append(depth_read(depths[i]))
        imgs_li.append(img_read(imgs[i]))
    depth_scale = 1000  # 暂定为100，因为深度值有40这种数字，在肠道内部，近距离看最近应该也是1-2cm左右
    cx = 319.5  # 396
    cy = 255.5  # 317
    fx = 389.93593
    fy = 309.77969
    point_3D_list = []
    XYZ1c = []
    colors = []
    # for depth, img, T in zip(depth_li, imgs_li, T_li):
    depth_0, img_0, T_0 = depth_li[0], imgs_li[0], T_li[0]
    point_3D_list_1 = []
    # 将内参写成一个矩阵
    instincts = np.asarray([389.93593, 0, 319.5, 0, 309.77969, 255.5, 0, 0, 1]).reshape(3, 3)
    depth_1, img_1, T_1 = depth_li[1], imgs_li[1], T_li[1]  # 需要提前知道下一个变换的位姿
    depth_2, img_2, T_2 = depth_li[2], imgs_li[2], T_li[2]
    point_3D_list_2 = []
    depth_error = 0
    img_error = 0
    num_projected_point = 0
    depth_error_list = []
    for u in range(0, depth_0.shape[0], 2):
        for v in range(0, depth_0.shape[1], 2):
            # a = depth_0[u, v]
            d = np.max(depth_0[u, v])
            if d == 0:
                continue  # 为0表示没有测量到
            point = []
            color = []
            zc = float(d)
            point.append((u - cx) * zc / fx)
            point.append((v - cy) * zc / fy)
            point.append(zc)
            point.append(1)
            point_3D = T_0 @ np.array(point).reshape(4, 1)
            point_3D_list_1.append(point_3D)

            # 反向投影
            point_camera = np.linalg.inv(T_1) @ point_3D  # 反投影时使用
            # point_camera = T_1@ point_3D
            point_camera = point_camera[0:3]
            pixel_coord = (1 / point_camera[2]) * instincts @ point_camera

            u_k = int(np.round(pixel_coord[0]))
            v_k = int(np.round(pixel_coord[1]))
            if v_k != v or u_k != u:
                print("v_k!=v")
            if u_k >= depth_0.shape[0] or u_k < 0 or v_k >= depth_0.shape[1] or v_k < 0:
                continue
            if float(img_1[u_k, v_k]) <= 5 or float(img_0[u, v]) <= 5:
                continue
            # if float(depth_1[u_k, v_k]) <= 20 or float(depth_0[u, v]) <= 20:
            #     continue
            # 这个投影下来的坐标和谁进行比较呢？而且应该要保证投影的坐标也是在像素平面内部的，不然就视作没有成功投影的像素点
            # 现在有两个像素坐标，就可以计算灰度值的差了
            # 还是说需要记录下这些新的uk vk 的值，作为成功投影的像素点，再取一个有效性mask ，去计算经过mask 屏蔽过后的两张图像之间的整体相似性作为loss呢
            print(point_camera[2] - np.max(depth_1[u, v]))
            num_projected_point += 1
            depth_error_list.append(np.abs(point_camera[2] - np.max(depth_1[u_k, v_k])))
            depth_error = depth_error + np.abs(point_camera[2] - np.max(depth_1[u_k, v_k]))
            img_error = img_error + np.abs(int(img_0[u, v]) - int(img_1[u_k, v_k]))

    show_2D(np.array(depth_error_list).reshape(len(depth_error_list), 1), "depth_error")
    print(T_0, T_1)
    print(f'深度值的范围:{np.min(depth_1)}到：{np.max(depth_1)}')
    print(f'成功投影的点数：{num_projected_point}，深度误差：{depth_error}，图像灰度值误差：{img_error}，'
          f'总的像素点数: {depth_0.shape[0] * depth_0.shape[1]}')
