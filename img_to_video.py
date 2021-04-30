import cv2
import os
from tqdm import tqdm
import re


def img_2_video(img_dir, img_shape=(640, 480), save_name="img_video", fps=5):
    """
    图片转视频, .mp4格式
    :param img_dir:
    :param img_shape:
    :param save_name:
    :param fps: 帧率
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_cat = cv2.VideoWriter(save_name + ".mp4", fourcc, fps, img_shape, True)  # 保存位置/格式

    total_img = os.listdir(img_dir)
    total_img = sorted(total_img, key=lambda x: int(re.findall('\d+', x)[0]))

    for img_name in tqdm(total_img):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        out_cat.write(image)  # 保存视频


if __name__ == '__main__':
    img_2_video('img')
