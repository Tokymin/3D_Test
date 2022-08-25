import csv
import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from path import Path
from scipy.spatial.transform import Rotation as R

from complete_loss_test.loss_functions import compute_errors, photo_and_geometry_loss, compute_smooth_loss


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
    depths = sorted(Path(("F:/Toky/Dataset/UnityCam/Recordings002/depth/")).files('*.exr'))
    imgs = sorted(Path(("F:/Toky/Dataset/UnityCam/Recordings002/photo/")).files('*.png'))

    with open(r"F:\Toky\Dataset\UnityCam\Recordings002/position_rotation.csv", encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        position_deltas.append(line[0:].split(" "))
    return imgs, depths, position_deltas


def depth_read(depth_img_file_name):
    depth_img = cv2.imread(depth_img_file_name, -1)  # , dtype=cv2.CV_32F
    # depth_img = np.transpose(depth_img, (2, 0, 1))
    q = np.asarray(depth_img, dtype=np.float64)[:, :, 2]
    return q * 100


def img_read(img_file_name):
    img = cv2.imread(img_file_name, 1)  # 直接读取灰度图像：0，直接读取RGB图像：1
    img = np.transpose(img, (2, 0, 1))  # 将h w c 0,1,2 变换为符合tensor习惯的c h w
    return np.asarray(img, dtype=np.float64)


# 这个脚本可以将像素平面的点根据内参，外参转换为3维空间中的点云
if __name__ == '__main__':
    imgs, depths, position_deltas = load_data()
    depth_li = []
    T_li = []
    imgs_li = []
    for i in range(3, 6):
        Rm = R.from_quat([position_deltas[i][3], position_deltas[i][4], position_deltas[i][5], position_deltas[i][6]])
        rotation_matrix = Rm.as_matrix()
        rvec = rotation_matrix  # 3*3,针对四元数的
        a = np.hstack((rvec, np.array([position_deltas[i][0], position_deltas[i][1], position_deltas[i][2]],
                                      dtype=np.float32).reshape(3, 1)))
        T_li.append(np.vstack((a, [0, 0, 0, 1])))  # 保持3*4的状态 ，若想要4*4 的旋转矩阵需 ： np.vstack((a, [0, 0, 0, 1]))
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
    depth_1, img_1, T_1 = depth_li[1], imgs_li[1], T_li[1]  # 需要提前知道下一个变换的位姿
    depth_2, img_2, T_2 = depth_li[2], imgs_li[2], T_li[2]

    point_3D_list_1 = []

    # 将内参写成一个矩阵
    instincts = np.asarray([389.93593, 0, 319.5, 0, 309.77969, 255.5, 0, 0, 1]).reshape(3, 3)

    point_3D_list_2 = []
    depth_error = 0
    img_error = 0
    num_projected_point = 0

    tgt_img = img_1
    ref_imgs = [torch.tensor(img_0).unsqueeze(0), torch.tensor(img_2).unsqueeze(0)]
    tgt_depth = depth_1
    ref_depths = [torch.tensor(depth_0).unsqueeze(0).unsqueeze(0), torch.tensor(depth_2).unsqueeze(0).unsqueeze(0)]

    # poses = [torch.tensor(np.linalg.inv(T_1)).unsqueeze(0), torch.tensor(T_2).unsqueeze(0)]
    # poses_inv = [torch.tensor(T_1).unsqueeze(0), torch.tensor(np.linalg.inv(T_2)).unsqueeze(0)]

    poses = [torch.tensor(T_2).unsqueeze(0), torch.tensor(np.linalg.inv(T_1)).unsqueeze(0)]
    poses_inv = [torch.tensor(np.linalg.inv(T_2)).unsqueeze(0), torch.tensor(T_1).unsqueeze(0)]

    loss_1, loss_2 = photo_and_geometry_loss(torch.tensor(tgt_img).unsqueeze(0), ref_imgs,
                                             torch.tensor(tgt_depth).unsqueeze(0).unsqueeze(0),
                                             ref_depths,
                                             torch.tensor(instincts).unsqueeze(0), poses,
                                             poses_inv)

    print(f'光度误差：{loss_1},几何投影误差：{loss_2}')
    loss_3 = compute_smooth_loss(torch.tensor(tgt_depth).unsqueeze(0).unsqueeze(0), torch.tensor(tgt_img).unsqueeze(0))
    print(f'深度的光滑度误差：{loss_3}')
