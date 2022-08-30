import csv
import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from path import Path
from scipy.spatial.transform import Rotation as R
from refence_v2_lyc import show_2D
from complete_loss_test.loss_functions import photo_and_geometry_loss, compute_smooth_loss

# cx = 160  # 396
# cy = 160  # 317
# fx = 157.549850
# fy = 156.3536121
# depth_scale = 100  # 暂定为100，因为深度值有40这种数字，在肠道内部，近距离看最近应该也是1-2cm左右
# instincts = np.asarray([157.549850, 0, 160, 0, 156.3536121, 160, 0, 0, 1]).reshape(3, 3)  # 将内参写成一个矩阵

cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0
instincts = np.asarray([518.0, 0, 325.5, 0, 519.0, 253.5, 0, 0, 1]).reshape(3, 3)


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
    depths = sorted(Path((r"C:\Users\DELL\Desktop\slambook2-master\ch5\rgbd\depth\\")).files('*.pgm'))
    imgs = sorted(Path((r"C:\Users\DELL\Desktop\slambook2-master\ch5/rgbd\color/")).files('*.png'))
    with open(r"C:\Users\DELL\Desktop\slambook2-master\ch5\rgbd/pose.txt",

              # depths = sorted(Path(("E:/Toky/dataSet/cd2rtzm23r-1/UnityCam/Colon/Pixelwise Depths/")).files('*.png'))
              # imgs = sorted(Path(("E:/Toky/dataSet/cd2rtzm23r-1/UnityCam/Colon/Frames/")).files('*.png'))
              # with open(r"E:/Toky/dataSet/cd2rtzm23r-1/UnityCam/Colon/Poses/colon_position_rotation.csv",
              encoding='utf-8') as file:  # _delta_processed
        content = file.readlines()
    for line in content:
        position_deltas.append(line[0:].split(" "))
    return imgs, depths, position_deltas


def depth_read(depth_img_file_name):
    depth_img = cv2.imread(depth_img_file_name, -1)  # , dtype=cv2.CV_32F
    # depth_img = np.transpose(depth_img, (2, 0, 1))
    q = np.asarray(depth_img, dtype=np.float64)
    # result = np.zeros_like(q)
    # result=cv2.normalize(q, result, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return q


def img_read(img_file_name):
    img = cv2.imread(img_file_name, 1)  # 直接读取灰度图像：0，直接读取RGB图像：1
    img = np.transpose(img, (2, 0, 1))  # 将h w c 0,1,2 变换为符合tensor习惯的c h w
    return np.asarray(img, dtype=np.float64)


if __name__ == '__main__':
    imgs, depths, position_deltas = load_data()
    depth_li = []
    T_li = []
    imgs_li = []
    loss_2_li = []

    for i in range(len(imgs)):
        Rm = R.from_quat([position_deltas[i][3], position_deltas[i][4], position_deltas[i][5], position_deltas[i][6]])
        rotation_matrix = Rm.as_matrix()
        rvec = rotation_matrix  # 3*3,针对四元数的
        a = np.hstack((rvec, np.array([position_deltas[i][0], position_deltas[i][1], position_deltas[i][2]],
                                      dtype=np.float32).reshape(3, 1)))
        T_li.append(np.vstack((a, [0, 0, 0, 1])))  # 保持3*4的状态 ，若想要4*4 的旋转矩阵需 ： np.vstack((a, [0, 0, 0, 1]))
        depth_li.append(depth_read(depths[i]))
        imgs_li.append(img_read(imgs[i]))

    for i in range(len(imgs_li) - 3):
        depth_0, img_0, T_0 = depth_li[i + 0], imgs_li[i + 0], T_li[i + 0]
        depth_1, img_1, T_1 = depth_li[i + 1], imgs_li[i + 1], T_li[i + 1]  # 需要提前知道下一个变换的位姿
        depth_2, img_2, T_2 = depth_li[i + 2], imgs_li[i + 2], T_li[i + 2]

        tgt_img = img_1
        ref_imgs = [torch.tensor(img_0).unsqueeze(0), torch.tensor(img_2).unsqueeze(0)]
        tgt_depth = depth_1
        ref_depths = [torch.tensor(depth_0).unsqueeze(0).unsqueeze(0),
                      torch.tensor(depth_2).unsqueeze(0).unsqueeze(0)]

        poses = [torch.tensor(np.linalg.inv(T_1)).unsqueeze(0), torch.tensor(T_2).unsqueeze(0)]
        poses_inv = [torch.tensor(T_1).unsqueeze(0), torch.tensor(np.linalg.inv(T_2)).unsqueeze(0)]

        loss_1, loss_2 = photo_and_geometry_loss(torch.tensor(tgt_img).unsqueeze(0), ref_imgs,
                                                 torch.tensor(tgt_depth).unsqueeze(0).unsqueeze(0),
                                                 ref_depths,
                                                 torch.tensor(instincts).unsqueeze(0), poses,
                                                 poses_inv)

        loss_2_li.append(loss_2)
        print(f'光度误差：{loss_1},几何投影误差：{loss_2}')
        loss_3 = compute_smooth_loss(torch.tensor(tgt_depth).unsqueeze(0).unsqueeze(0),
                                     torch.tensor(tgt_img).unsqueeze(0))

        print(f'深度的光滑度误差：{loss_3}')

    show_2D(np.array(loss_2_li).reshape(len(loss_2_li), 1), "geometry_loss")
