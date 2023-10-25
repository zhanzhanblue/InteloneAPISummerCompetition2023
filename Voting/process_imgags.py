import os
from PIL import Image

save_path = 'datasave/'
new_save_path = 'newdatasave/'

img_files = os.listdir(save_path)


def find_left_upper_right_lower(_image):
    """这里是找到图片的数字的左上右下的位置，便于图片切割"""
    N = _image.height
    # 先定义两个桶
    x_bucket = [0 for _ in range(N)]
    y_bucket = [0 for _ in range(N)]

    # 遍历找到黑点位置
    for x in range(_image.width):
        for y in range(_image.height):
            if not _image.getpixel((x, y)):
                x_bucket[x] = True
                y_bucket[y] = True

    try:
        left = x_bucket.index(1)
    except:
        left = 0
    try:
        upper = y_bucket.index(1)
    except:
        upper = 0
    try:
        right = 99 - x_bucket[::-1].index(1)
    except:
        right = _image.width
    try:
        lower = 99 - y_bucket[::-1].index(1)
    except:
        lower = _image.height

    # 图片显示的不是很对劲，可以增加一部分像素
    left = max(left - 10, 0)
    upper = max(upper - 10, 0)
    right = min(right + 10, 99)
    lower = min(lower + 10, 99)

    return left, upper, right, lower


def change0_1(_image):
    """ 把图片变成黑白的，当 pixel=155 为分界线的时候非常优 0 for black"""
    for x in range(_image.width):
        for y in range(_image.height):
            if _image.getpixel((x, y)) < 155:
                _image.putpixel((x, y), 0)
            else:
                _image.putpixel((x, y), 255)
    return _image


def process_image(_image):
    """这里要黑白，裁切图片了哦"""
    _image = change0_1(_image)
    _image = _image.crop(find_left_upper_right_lower(_image))
    _image = _image.resize((28, 28))  # 把图片再变回去啊
    # _image = change0_1(_image)  # 看看这里时候需要再用黑白变回去
    return _image


def image_process():
    """处理图片专用，一会还要用process处理"""

    # 所有图片
    image_files = os.listdir(save_path)

    # 改图片，将图片变成new_data_save
    for image_file in image_files:
        image_ = Image.open(save_path + image_file)
        image_new = process_image(image_)
        image_new.save(new_save_path + image_file)


def test():
    """测试专用"""
    a = [0, 1, 2, 3, 4]
    b = [4, 5, 56, 6]

    a += b
    print(a)

    exit(0)

    image_ = Image.open(save_path + '0-1无名氏1.jpg')

    for x in range(image_.width):
        for y in range(image_.height):
            print(type(image_.getpixel((x, y))))

    # image_ = change0_1(image_)
    # print(find_left_upper_right_lower(image_))
    # # exit(0)
    # image_ = image_.crop(find_left_upper_right_lower(image_))
    image_ = image_.resize((1000, 32))
    image_.save('234.jpg')


if __name__ == '__main__':
    image_process()
    # test()
    # find_left_upper_right_lower()
    pass
