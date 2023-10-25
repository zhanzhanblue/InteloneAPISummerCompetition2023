import re

import numpy as np
from PIL import Image
import os.path
import struct

data_path = 'data/'
save_path = 'newdatasave/'
train_labels_file = 'train-labels-idx1-ubyte.txt/'
train_images_file = 'train-images-idx3-ubyte/'
test_labels_file = 't10k-labels-idx1-ubyte.txt/'
test_images_file = 't10k-images-idx3-ubyte/'
HWDG_train_labels_file = 'HWDG/train-labels-idx1-ubyte'
HWDG_train_images_file = 'HWDG/train-images-idx3-ubyte'
HWDG_test_labels_file = 'HWDG/t10k-labels-idx1-ubyte'
HWDG_test_images_file = 'HWDG/t10k-images-idx3-ubyte'
MNIST_train_labels_file = 'MNIST/train-labels-idx1-ubyte'
MNIST_train_images_file = 'MNIST/train-images-idx3-ubyte'
MNIST_test_labels_file = 'MNIST/t10k-labels-idx1-ubyte'
MNIST_test_images_file = 'MNIST/t10k-images-idx3-ubyte'


def trans1():
    # 所有图片
    img_files = os.listdir(data_path)

    # 将图片转化成灰度值表示
    for img_file in img_files:
        # 正则表达式提取有用的信息
        # print(img_file)
        x = re.match(r'(.*)-(\d).*', img_file)
        name = x.group(1)
        label = x.group(2)
        # print(x.group(1), x.group(2))

        # 转化灰度图像
        image_ = Image.open(data_path + img_file)
        image_new = image_.convert('L')
        # 像素大小 100x100

        image_new = image_new.resize((100, 100))

        # 保存图片
        image_new.save(save_path + label + '-' + name + '.jpg')


def trans2():
    # 分出HWDG的四个文件

    # 所有图片
    img_files = os.listdir(save_path)

    # 个数
    n = len(img_files)  # 630 个
    # print(n)
    # 每一种是63个人
    per = n // 10  # 63个
    # print(per)

    # 分出的训练集和测试集的图片是不能有交集的 500:130 大约是 8:2
    # 可以在图片中挑两个作为测试集，剩余在训练集中,一个人10张，13个人130张
    # 第 1,3,5,7,9,11,13,15,17,19,21,23,25 个人 (这有第 0 个人的)

    test_people = list(range(1, 26, 2))


    train_people = []
    for person in range(0, 63):
        if person not in test_people:
            train_people.append(person)
    # print(train_people)

    for person in test_people:
        test_list = img_files[person::63]
        # 放入t10k images 中
        for img_file in test_list:
            image_ = Image.open(save_path + img_file)
            image_.save(test_images_file + img_file)
            # print(img_file)

    # 将剩下的放在train中
    for person in train_people:
        train_list = img_files[person::63]
        # 放入train images 中
        for img_file in train_list:
            image_ = Image.open(save_path + img_file)
            image_.save(train_images_file + img_file)


def trans3_images():
    """这里要和MNIST数据集images进行对照一下，看看怎么把它搞进去"""
    # 图片是100x100的， 10000个pixel

    # 写train-images-idx3-ubyte文件
    # 1. magic number 32 位 2051, images number 500
    with open(HWDG_train_images_file, 'wb') as f:
        # magic number images number rows number colmun number
        s = struct.pack('>iiii', 2051, 500, 100, 100)
        f.write(s)

    # 将所有图片的像素（pixel）加入到这个文件中
    img_list = os.listdir(train_images_file)
    with open(HWDG_train_images_file, 'ab') as f:
        for img_file in img_list:
            image_ = Image.open(train_images_file + img_file)
            # print(image_)
            for x in range(image_.height):
                for y in range(image_.width):
                    pixel = image_.getpixel((x, y))
                    s = struct.pack('>B', pixel)
                    f.write(s)

    # 写t10k-images-idx3-ubyte文件
    # magic number 32 位 2051, images number 130
    with open(HWDG_test_images_file, 'wb') as f:
        # magic number images number rows number columns number
        s = struct.pack('>iiii', 2051, 130, 100, 100)
        f.write(s)

    # 将所有图片的像素（pixel）加入到这个文件中
    img_list = os.listdir(test_images_file)
    with open(HWDG_test_images_file, 'ab') as f:
        for img_file in img_list:
            image_ = Image.open(test_images_file + img_file)
            for x in range(image_.height):
                for y in range(image_.width):
                    pixel = image_.getpixel((x, y))
                    s = struct.pack('>B', pixel)
                    f.write(s)


def trans3_labels():
    """这里要和MNIST数据集lables进行对照一下，看看怎么把它搞进去"""

    # 写train-labels-idx1-ubyte文件
    # 1. magic number 32 位 2049, images number 500
    with open(HWDG_train_labels_file, 'wb') as f:
        # magic number images number rows number columns number
        s = struct.pack('>ii', 2049, 500)
        f.write(s)

    # 将train_images_idx3_ubyte的标签(文件名的第一个)写入HWDG中
    img_files = os.listdir(train_images_file)

    # 遍历所有图片，用正则表达式提取出来
    with open(HWDG_train_labels_file, 'ab') as f:
        for img_file in img_files:
            # print(img_file)
            x = re.match(r'(\d)-.*', img_file)
            # print(x)
            # print(x.group(1))

            number = x.group(1)
            # 将这个提取出来的number放在HWDG中
            s = struct.pack('>B', int(number))
            f.write(s)

    # 写t10k-lables-idx1-ubyte 二进制文件
    # magic number 32 位 2049, images number 130
    with open(HWDG_test_labels_file, 'wb') as f:
        # magic number images number
        s = struct.pack('>ii', 2049, 130)
        f.write(s)

    # 将t10k_images_idx3_ubyte的标签(文件名的第一个)写入HWDG中
    img_files = os.listdir(test_images_file)

    # 遍历所有图片，用正则表达式提取出来
    with open(HWDG_test_labels_file, 'ab') as f:
        for img_file in img_files:
            # print(img_file)
            x = re.match(r'(\d)-.*', img_file)
            # print(x)
            # print(x.group(1))

            number = x.group(1)
            # 将这个提取出来的number放在HWDG中
            s = struct.pack('>B', int(number))
            f.write(s)


def load_HWDG_train():
    # 读labels
    with open(HWDG_train_labels_file, 'rb') as f:
        magic_number, items = struct.unpack('>ii', f.read(8))
        print('lables : magic number = {}, items number = {} '.format(magic_number, items))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # 读照片信息
    with open(HWDG_train_images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>iiii', f.read(16))
        print('images : magic number = {}, items number = {}, rows = {}, columns = {} '.format(
            magic_number, items, rows, cols
        ))
        images = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def load_HWDG_test():
    # 读labels
    with open(HWDG_test_labels_file, 'rb') as f:
        magic_number, items = struct.unpack('>ii', f.read(8))
        print('lables : magic number = {}, items number = {} '.format(magic_number, items))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # 读照片信息
    with open(HWDG_test_images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>iiii', f.read(16))
        print('images : magic number = {}, items number = {}, rows = {}, columns = {} '.format(
            magic_number, items, rows, cols
        ))
        images = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def load_MNIST_train():
    # 读labels
    with open(MNIST_train_labels_file, 'rb') as f:
        magic_number, items = struct.unpack('>ii', f.read(8))
        print('lables : magic number = {}, items number = {} '.format(magic_number, items))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # 读照片信息
    with open(MNIST_train_images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>iiii', f.read(16))
        print('images : magic number = {}, items number = {}, rows = {}, columns = {} '.format(
            magic_number, items, rows, cols
        ))
        images = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def load_MNIST_test():
    # 读labels
    with open(MNIST_test_labels_file, 'rb') as f:
        magic_number, items = struct.unpack('>ii', f.read(8))
        print('lables : magic number = {}, items number = {} '.format(magic_number, items))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    # 读照片信息
    with open(MNIST_test_images_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>iiii', f.read(16))
        print('images : magic number = {}, items number = {}, rows = {}, columns = {} '.format(
            magic_number, items, rows, cols
        ))
        images = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


if __name__ == '__main__':
    # 灰度、换名、加一个标准化
    # trans1()

    # 分出四个文件
    trans2()

    # 将这些数据集变成二进制放到和MNIST数据集一样的文件中
    trans3_images()
    trans3_labels()
    pass