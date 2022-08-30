# 矩阵相乘

import numpy as np


def matrix_mutual(m1, m2):
    result = np.dot(m1, m2)

    return result


# -*- coding: utf-8 -*-
import os


def rename_for_single(a):
    ext_name = '.png'

    if a < 10:
        new_name = "0000" + str(a) + ext_name
    elif a < 100:
        new_name = "000" + str(a) + ext_name
    elif a < 1000:
        new_name = "00" + str(a) + ext_name
    elif a < 10000:
        new_name = "0" + str(a) + ext_name
    elif a < 100000:
        new_name = "" + str(a) + ext_name
    # os.rename(os.path.join(path, file), os.path.join(path, new_name))
    return new_name


def rename():
    a = 0
    ext_name = '.png'
    for file in sorted_filenames:
        if a < 10:
            new_name = "0000" + str(a) + ext_name
        elif a < 100:
            new_name = "000" + str(a) + ext_name
        elif a < 1000:
            new_name = "00" + str(a) + ext_name
        elif a < 10000:
            new_name = "0" + str(a) + ext_name
        elif a < 100000:
            new_name = "" + str(a) + ext_name
        # os.rename(os.path.join(path, file), os.path.join(path, new_name))
        os.rename(os.path.join(path, file), os.path.join(path, new_name))
        a += 1


def mysort(path):
    filelists = os.listdir(path)
    sort_num_first = []
    for file in filelists:
        f = file.split(".")[0]
        g = f.split("_")[2]
        # sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.append(int(g))
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in filelists:
            if str(sort_num) == file.split(".")[0]:
                sorted_file.append(file)
    return sorted_file


def depth_sort(path):
    # 针对形如 aov_image_0500 格式的深度图片
    filelists = os.listdir(path)
    sort_num_first = []
    for file in filelists:
        f = file.split(".")[0]
        g = f.split("_")[2]
        # sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.append(int(g))
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in filelists:
            if sort_num == int(file.split(".")[0].split("_")[2]):
                sorted_file.append(file)
    return sorted_file


def depth_sort_endo(path):
    # 针对形如 frame_000044.jpg 格式的深度图片
    filelists = os.listdir(path)
    sort_num_first = []
    for file in filelists:
        f = file.split(".")[0]
        g = f.split("_")[1]
        # sort_num_first.append(int(file.split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
        sort_num_first.append(int(g))
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first:
        for file in filelists:
            if sort_num == int(file.split(".")[0].split("_")[1]):
                sorted_file.append(file)
    return sorted_file


def get_filename(file_dir):
    filename = None
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        filename = files
    return filename


import os
import cv2
import numpy as np
import concurrent.futures

'''
    multi-process to crop pictures.
'''


def crop(file_path_list):
    origin_path, save_path = file_path_list
    img = cv2.imread(origin_path)
    a = np.where(img > 1, img, 0 * img)  # 过滤掉那些0,0,1的像素点
    cv2.imwrite(save_path, a)


def multi_process_crop(input_dir):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(crop, input_dir)


# 针对训练集
def make_af_dataset_file():
    train_partition = 160  # 写入train.txt
    partition = 200  # 隔160个连续图像对后取40个连续图像对进行测试。写入test.txt
    file_handle_train = open(r'F:\Toky\PythonProject\3D_Test\af_dataset\train.txt',
                             mode='a+')  # 这里的key对应的是原list里首张图片的下标索引，另外一张图就是索引加上step，用来找到对应的图片
    file_handle_test = open(r'F:\Toky\PythonProject\3D_Test\af_dataset\test.txt',
                            mode='a+')  # 这里的key对应的是原list里首张图片的下标索引，另外一张图就是索引加上step，用来找到对应的图片
    count = 0
    alllist = os.listdir(u"F:\Toky\Dataset\Endo_colon_unity\\")  # 组装scence文件夹的名字
    max_index = len(alllist) - 2
    iters = range(0, int(max_index / partition) + 2)
    j = 0
    for i in range(0, max_index):
        if count >= max_index:
            break
        else:
            iter = iters[j]
            if i != 0 and i % partition == 0:
                j = j + 1
            if i == 0 or i / (iter * partition + train_partition) < 1:  # 第iter轮,<1 说明属于训练集
                # path + image_file_index_1
                file_handle_train.write('train_dataset/photo/' + str(count) + '\n')
                count += 1
            else:  # <1 说明属于测试集
                file_handle_test.write('train_dataset/photo/' + str(count) + '\n')
                count += 1
    file_handle_train.close()
    file_handle_test.close()


def make_sc_dataset_file():
    train_partition = 160  # 写入train.txt
    partition = 200  # 隔160个连续图像对后取40个连续图像对进行测试。写入test.txt
    file_handle_train = open(r'F:\Toky\PythonProject\3D_Test\sc_dataset\train.txt',
                             mode='a+')
    file_handle_test = open(r'F:\Toky\PythonProject\3D_Test\sc_dataset\test.txt',
                            mode='a+')
    count = 0
    alllist = os.listdir(u"F:\Toky\Dataset\Endo_colon_unity\photo\\")  #
    max_index = len(alllist) - 2
    iters = range(0, int(max_index / partition) + 2)
    j = 0
    for i in range(0, max_index):
        if count >= max_index:
            break
        else:
            iter = iters[j]
            if i != 0 and i % partition == 0:
                j = j + 1
            if i == 0 or i / (iter * partition + train_partition) < 1:  # 第iter轮,<1 说明属于训练集
                # path + image_file_index_1
                file_handle_train.write('photo/' + str(rename_for_single(count)) + '\t' + "l" + '\n')
                count += 1
            else:  # <1 说明属于测试集
                file_handle_test.write('photo/' + str(rename_for_single(count)) + '\t' + "l" + '\n')
                count += 1
    file_handle_train.close()
    file_handle_test.close()


def make_af_test_dataset_file():
    train_partition = 160  # 写入train.txt
    partition = 200  # 隔160个连续图像对后取40个连续图像对进行测试。写入test.txt
    count = 0
    alllist = os.listdir(u"F:\Toky\Dataset\Endo_colon_unity\photo\\")  #
    max_index = len(alllist) - 2
    iters = range(0, int(max_index / partition) + 2)
    j = 0
    for i in range(0, max_index):
        if count >= max_index:
            break
        else:
            iter = iters[j]
            if i != 0 and i % partition == 0:
                j = j + 1
            if i == 0 or i / (iter * partition + train_partition) < 1:  # 第iter轮,<1 说明属于训练集
                continue  # 属于测试集的部分直接跳过
                count += 1
            else:  # <1 说明属于测试集
                file_handle_test = open(
                    r'F:\Toky\PythonProject\3D_Test\af_dataset\test\test' + str(iter) + '.txt',
                    mode='a+')
                file_handle_test.write('photo/' + str(rename_for_single(count)) + '\t' + "l")
                count += 1
    file_handle_test.close()


def make_af_gt_pose_dataset_file():
    train_partition = 160  # 写入train.txt
    partition = 200  # 隔160个连续图像对后取40个连续图像对进行测试。写入test.txt
    count = 0

    gt_kitti_path = 'F:\Toky\Dataset\Endo_colon_unity\colon_position_rotation_kitti.txt'
    with open(gt_kitti_path, encoding='utf-8') as file:
        content = file.readlines()
    max_index = len(content) - 2
    iters = range(0, int(max_index / partition) + 2)
    j = 0
    for i, line in zip(range(0, max_index), content):

        if count >= max_index:
            break
        else:
            iter = iters[j]
            if i != 0 and i % partition == 0:
                j = j + 1
            if i == 0 or i / (iter * partition + train_partition) < 1:  # 第iter轮,<1 说明属于训练集
                continue  # 属于测试集的部分直接跳过
                count += 1
            else:  # <1 说明属于测试集
                file_handle_test = open(
                    r'F:\Toky\PythonProject\3D_Test\af_dataset\gt_pose\gt_pose' + str(iter) + '.txt',
                    mode='a+')
                file_handle_test.write(line)
                count += 1
    file_handle_test.close()


if __name__ == '__main__':
    # 1 矩阵相乘
    # A = np.asarray([157.549850, 0, 160, 0, 156.3536121, 160, 0, 0, 1]).reshape(3, 3)  # 将内参写成一个矩阵
    #
    # B = np.array([[1, 0, 0, -1.8324],
    #               [0, 1, 0, -9.1007],
    #               [0, 0, 1, 2.3064]])
    #
    # result = matrix_mutual(A, B)
    # print(result)

    # 2 重命名
    #
    # ext_name = ".png"
    # path = r'F:\Toky\Dataset\Endo_colon_unity\depth\\'
    # sorted_filenames = depth_sort(path)
    # # sorted_filenames = os.listdir(path)  # 不用排序的，以数字开头的
    # rename()

    # 3 去除黑边
    # data_dir = 'I:\dataset\endoslam_unity_colon\Frames\\'
    # save_dir = 'F:\Toky\Dataset\Endo_colon_unity\\'
    # path_list = [(os.path.join(data_dir, o), os.path.join(save_dir, o)) for o in os.listdir(data_dir)]
    # start = time.time()
    # multi_process_crop(path_list)
    # print(f'Total cost {time.time()-start} seconds')

    # 4 划分数据集
    # make_sc_dataset_file()
    make_af_test_dataset_file()
