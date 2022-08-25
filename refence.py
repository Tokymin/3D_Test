import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from path import Path
from scipy.spatial.transform import Rotation as R


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
    depths = sorted(Path(("F:/Toky/Dataset/UnityCam/Recordings001/depth/")).files('*.pgm'))
    imgs = sorted(Path(("F:/Toky/Dataset/UnityCam/Recordings001/photo/")).files('*.png'))

    with open(r"F:\Toky\Dataset\UnityCam\Recordings001/position_rotation.csv", encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        position_deltas.append(line[0:].split(" "))
    return imgs, depths, position_deltas


def depth_read(depth_img_file_name):
    depth_img = cv2.imread(depth_img_file_name, 0)  # , dtype=cv2.CV_32F
    return depth_img


def img_read(img_file_name):
    depth_img = cv2.imread(img_file_name, cv2.IMREAD_UNCHANGED)  # , dtype=cv2.CV_32F
    return depth_img


# 这个脚本可以将像素平面的点根据内参，外参转换为3维空间中的点云
if __name__ == '__main__':
    imgs, depths, position_deltas = load_data()
    depth_li = []
    T_li = []
    imgs_li = []
    for i in range(0, 5):
        Rm = R.from_quat([position_deltas[i][3], position_deltas[i][4], position_deltas[i][5], position_deltas[i][6]])
        rotation_matrix = Rm.as_matrix()
        rvec = rotation_matrix  # 3*3,针对四元数的
        a = np.hstack((rvec, np.array([position_deltas[i][0], position_deltas[i][1], position_deltas[i][2]],
                                      dtype=np.float32).reshape(3, 1)))
        T_li.append(a)
        depth_li.append(depth_read(depths[i]))
        imgs_li.append(img_read(imgs[i]))

    depth_scale = 1000  # 暂定为100，因为深度值有40这种数字，在肠道内部，近距离看最近应该也是1-2cm左右
    cx = 178.5604
    cy = 181.8043
    fx = 156.0418
    fy = 178.5604
    point_3D_list = []
    XYZ1c = []
    colors = []
    for depth, img, T in zip(depth_li, imgs_li, T_li):
        b, g, r, _ = cv2.split(img)
        for u in range(0, depth.shape[0], 8):
            for v in range(0, depth.shape[1], 8):
                d = depth[u][v]
                if d == 0:
                    continue  # 为0表示没有测量到
                point = []
                color = []
                zc = float(d) / depth_scale
                point.append((u - cx) * zc / fx)
                point.append((v - cy) * zc / fy)
                point.append(zc)
                point.append(1)
                point_3D = T @ np.array(point).reshape(4, 1)
                point_3D_list.append(point_3D)
                colors.append(color)
    show(np.array(point_3D_list), "")
