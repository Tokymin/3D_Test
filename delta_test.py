import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    depths = sorted(Path(("E:/Toky/dataSet/cd2rtzm23r-1/UnityCam/Colon/Pixelwise Depths/")).files('*.png'))
    imgs = sorted(Path(("E:/Toky/dataSet/cd2rtzm23r-1/UnityCam/Colon/Frames/")).files('*.png'))
    # with open(r"E:/Toky/dataSet/cd2rtzm23r-1/UnityCam/Colon/Poses/colon_position_rotation.csv",
    with open(r"F:/Toky/Dataset/UnityCam/Recordings003/position_rotation.csv",
              encoding='utf-8') as file:
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

depth_scale = 1000  # 暂定为100，因为深度值有40这种数字，在肠道内部，近距离看最近应该也是1-2cm左右
cx = 160  # 396
cy = 160  # 317
fx = 157.549850
fy = 156.3536121
instincts = np.asarray([157.549850, 0, 160, 0, 156.3536121, 160, 0, 0, 1]).reshape(3, 3)  # 将内参写成一个矩阵

if __name__ == '__main__':
    imgs, depths, position_deltas = load_data()
    depth_li = []
    T_li = []
    imgs_li = []
    for i in range(len(position_deltas)-1):
        Rm = R.from_quat([position_deltas[i][3], position_deltas[i][4], position_deltas[i][5], position_deltas[i][6]])
        rotation_matrix = Rm.as_matrix()
        rvec = rotation_matrix  # 3*3,针对四元数的
        a = np.hstack((rvec,
                       np.array([position_deltas[i][0], position_deltas[i][1], position_deltas[i][2]],
                                dtype=np.float32).reshape(3, 1)))
        T_li.append(np.vstack(
            (a, [0, 0, 0, 1])))
        depth_li.append(depth_read(depths[i]))
        imgs_li.append(img_read(imgs[i]))

    point_3D_list = []
    XYZ1c = []
    colors = []
    for i in range(len(imgs_li) - 3):
        depth_0, img_0, T_0 = depth_li[0 + i], imgs_li[0 + i], T_li[0 + i]
        depth_1, img_1, T_1 = depth_li[1 + i], imgs_li[1 + i], T_li[1 + i]  # 需要提前知道下一个变换的位姿
        depth_2, img_2, T_2 = depth_li[2 + i], imgs_li[2 + i], T_li[2 + i]

        point_3D_list_1 = []
        point_3D_list_2 = []
        depth_error = 0
        img_error = 0
        num_projected_point = 0
        depth_error_list = []
        for u in range(0, depth_0.shape[0], 2):
            for v in range(0, depth_0.shape[1], 2):
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
                point_camera = point_camera[0:3]
                pixel_coord = (1 / point_camera[2]) * instincts @ point_camera

                u_k = int(np.round(pixel_coord[0]))
                v_k = int(np.round(pixel_coord[1]))
                # if v_k != v or u_k != u:
                #     print("v_k!=v")
                if u_k >= depth_0.shape[0] or u_k < 0 or v_k >= depth_0.shape[1] or v_k < 0:
                    continue
                if float(img_1[u_k, v_k]) <= 5 or float(img_0[u, v]) <= 5:
                    continue
                # if float(depth_1[u_k, v_k]) <= 20 or float(depth_0[u, v]) <= 20:
                #     continue
                num_projected_point += 1
                depth_error_list.append(np.abs(point_camera[2] - np.max(depth_1[u_k, v_k])))
                depth_error = depth_error + np.abs(point_camera[2] - np.max(depth_1[u_k, v_k]))
                img_error = img_error + np.abs(int(img_0[u, v]) - int(img_1[u_k, v_k]))

        # show_2D(np.array(depth_error_list).reshape(len(depth_error_list), 1), "depth_error")
        # print(T_0, T_1)
        print(f'深度值的范围:{np.min(depth_1)}到：{np.max(depth_1)}，平均值是：{np.mean(depth_1)}')
        print(
            f'成功投影的点数：{num_projected_point}，深度误差：{depth_error}，深度误差均值：{depth_error/num_projected_point},图像灰度值误差：{img_error}，'
            f'总的像素点数: {depth_0.shape[0] * depth_0.shape[1]}')
