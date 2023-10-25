import struct
import numpy as np
from PIL import Image
from sklearn import preprocessing

new_save_path = 'newdatasave/'
HWDG_train_labels_file = 'HWDG/train-labels-idx1-ubyte'
HWDG_train_images_file = 'HWDG/train-images-idx3-ubyte'
HWDG_test_labels_file = 'HWDG/t10k-labels-idx1-ubyte'
HWDG_test_images_file = 'HWDG/t10k-images-idx3-ubyte'


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
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(500, 28 * 28)

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
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(130, 28 * 28)

    return images, labels


# train_data_HWDG, train_labels_HWDG = load_HWDG_train()
# test_data_HWDG, test_labels_HWDG = load_HWDG_test()


# 提取特征

# 1、笔画密度特征 8个
def strokes_density(_image: Image):
    """传参是一张图片， 目的是将笔画的密度特征提取出来，共有8个特征"""
    # 水平方向是每隔7个提取一次，分别是0,7,14,21
    # 垂直方向是每隔7个提取一次，分别是0,7,14,21

    extract_array = [0, 7, 14, 21]

    res = []  # 所有特征

    # 首先是x方向
    for x in extract_array:
        _sum = 0
        for y in range(_image.height):
            if _image.getpixel((x, y)) == 0:  # 黑点
                _sum += 1
        res.append(_sum)

    # 然后是y方向
    for y in extract_array:
        _sum = 0
        for x in range(_image.width):
            if _image.getpixel((x, y)) == 0:  # 黑点
                _sum += 1
        res.append(_sum)

    return res


# 2、轮廓特征 14个
def contour(_image: Image):
    """得到图片的额轮廓特征，总共是14个特征"""

    N = _image.height
    # 定义几个东西
    # 1 Lp(k)、Rp(k) 第k行的左右轮廓
    # 2 Up(k)、Dp(k) 第k列的左右轮廓
    # 3 W(k) 字符宽度
    # 4 H(k) 字符高度

    Lp = [0 for _ in range(N)]
    Rp = [0 for _ in range(N)]
    Up = [0 for _ in range(N)]
    Dp = [0 for _ in range(N)]
    W = [0 for _ in range(N)]
    H = [0 for _ in range(N)]
    Ldif = [0 for _ in range(N - 1)]  # 左边线的一阶有限差
    Rdif = [0 for _ in range(N - 1)]  # 右边线的一阶有限差

    # 1
    for y in range(N):
        # 定义个桶
        bucket = [0 for _ in range(N)]
        for x in range(N):
            if _image.getpixel((x, y)) == 0:
                bucket[x] = 1
        try:
            Lp[y] = bucket.index(1)
        except:
            Lp[y] = 0
        try:
            Rp[y] = N - bucket[::-1].index(1)
        except:
            Rp[y] = N - 1

    # 2
    for x in range(N):
        # 定义一样的桶
        bucket = [0 for _ in range(N)]
        for y in range(N):
            if _image.getpixel((x, y)) == 0:
                bucket[y] = 1
        try:
            Up[x] = bucket.index(1)
        except:
            Up[x] = 0
        try:
            Dp[x] = N - bucket[::-1].index(1)
        except:
            Dp[x] = N - 1

    # 3
    for x in range(N):
        W[x] = Rp[x] - Lp[x]

    # 4
    for x in range(N):
        H[x] = Dp[x] - Up[x]

    # 这个是比率
    ratio = N / max(W)

    width_height_ratio = max(W) / max(H)

    for k in range(N - 1):
        Ldif[k] = Lp[k + 1] - Lp[k]
        Rdif[k] = Rp[k + 1] - Rp[k]

    Lpeak_positive = max(Ldif)
    Rpeak_positive = max(Rdif)
    Lpeak_negative = min(Ldif)
    Rpeak_negative = min(Rdif)
    Lpeak = abs(Lpeak_positive) + abs(Lpeak_negative)
    Rpeak = abs(Rpeak_positive) + abs(Rpeak_negative)

    # 有效宽度、字符比率、有效高度、宽高比、轮廓左侧最大值、轮廓左侧最小智、右侧最大值、右侧最小值、
    # 边缘线左侧正峰值、边缘线左侧负峰值、右侧正峰值、右侧负峰值、左侧正负峰值之和、右侧正负峰值之和
    return max(W), ratio, max(H), width_height_ratio, max(Lp), min(Lp), max(Rp), min(
        Rp), Lpeak_positive, Lpeak_negative, Rpeak_positive, Rpeak_negative, Lpeak, Rpeak


# 3、投影特征 2个
def projection(_image: Image):
    """这里是图片的投影特征，可以得到2个投影，分别从不同的方向看"""

    N = _image.height  # 就是N * N的大小

    # 桶又来了
    up = [0 for _ in range(N)]
    left = [0 for _ in range(N)]

    for x in range(N):
        for y in range(N):
            if _image.getpixel((x, y)) == 0:
                up[x] = 1
                left[y] = 1

    # 两个方向的投影
    return sum(up), sum(left)


# 4、重心的问题 3个
def gravity(_image: Image):
    """这里得到图片的重心的特征，有3个特征，先看看是不是"""

    N = _image.height

    # 先定义两个东西, 重心的位置
    m_bar = 0
    n_bar = 0

    sum_image = 0
    for x in range(N):
        for y in range(N):
            if _image.getpixel((x, y)) == 0:
                sum_image += 1

    if sum_image == 0:
        sum_image = 1

    for n in range(N):
        for m in range(N):
            m_bar += _image.getpixel((m, n)) * m / sum_image
            n_bar += _image.getpixel((m, n)) * n / sum_image

    # 重心距
    L = 0
    for n in range(N):
        for m in range(N):
            L += (m - m_bar) * (n - n_bar)

    # 重心的位置，重心距
    return m_bar, n_bar, L


# 5、首个黑点的位置 4个
def first_dot_location(_image: Image):
    """这里是找黑点的特征，总共是4个，分别是四个方向"""

    N = _image.height

    # 中心位置
    centre_x = N // 2
    center_y = N // 2

    # 向四个方向延伸

    # up
    up = 0
    for y in range(center_y, 0, -1):
        if _image.getpixel((centre_x, y)) == 0:
            up = y
            break
    # down
    down = N - 1
    for y in range(center_y, N):
        if _image.getpixel((centre_x, y)) == 0:
            down = y
            break
    # left
    left = 0
    for x in range(centre_x, 0, -1):
        if _image.getpixel((x, center_y)) == 0:
            left = x
            break
    # right
    right = N - 1
    for x in range(centre_x, N):
        if _image.getpixel((x, center_y)) == 0:
            right = x
            break

    # 返回上下左右四个方向首个黑点的位置
    return up, down, left, right


# 6、粗网格密度，分成四个象限4个特征
def grid_density(_image: Image):
    """这里要把图片分四个象限，得到4个特征"""

    # 可以用中心来分
    N = _image.height
    center_x = N // 2
    center_y = N // 2

    # 先初始化一波
    first_quadrant = 0
    second_quadrant = 0
    third_quadrant = 0
    forth_quadrant = 0

    # 第一象限
    for x in range(center_x):
        for y in range(center_y):
            first_quadrant += _image.getpixel((x, y)) == 0

    for x in range(center_x, N):
        for y in range(center_y):
            second_quadrant += _image.getpixel((x, y)) == 0

    for x in range(center_x):
        for y in range(center_y, N):
            third_quadrant += _image.getpixel((x, y)) == 0

    for x in range(center_x, N):
        for y in range(center_y, N):
            forth_quadrant += _image.getpixel((x, y)) == 0

    # 返回第1、2、3、4 象限的密度
    return first_quadrant, second_quadrant, third_quadrant, forth_quadrant


def get_features():
    def process(data):
        # 正则化
        data = preprocessing.normalize(data)
        # 标准化
        data = preprocessing.scale(data)
        return data
    res = []
    res.append(process(strokes_density()))



def test():
    """测试专用"""
    image_ = Image.open(new_save_path + '0-1无名氏1.jpg')
    # import process_imgags
    # image_ = process_imgags.change0_1(image_)
    # image_.show()
    for x in range(image_.width):
        for y in range(image_.height):
            print(image_.getpixel((x, y)), end=' ')
        print()
    print(grid_density(image_))


if __name__ == '__main__':
    test()
