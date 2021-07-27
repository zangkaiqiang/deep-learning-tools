"""
图像文件的批量切割
"""
import os
import cv2
import tqdm
from annotation import img_cut
import argparse


def cut1010():
    """
    切割1010数据
    :return:
    """
    images_name = os.listdir('data/1010')
    images_path = [os.path.join('data/1010', i) for i in images_name]

    if os.path.isdir('output/cut_result') is False:
        os.makedirs('output/cut_result')

    for image_path, image_name in zip(images_path, images_name):
        img = cv2.imread(image_path)
        imgs = img_cut(img, 400, 400, 400)
        for index, i in enumerate(imgs):
            cv2.imwrite('output/cut_result/%s-%.4d.jpg' % (image_name[:-4], index), i)


def cut4helc3_1():
    """
    切割data/1010/13-COPD-F-003-1.png
    该图像是helc3使用的原图
    将其切割成小图建立项目
    :return:
    """
    img = cv2.imread('data/1010/13-COPD-F-003-1.png')
    imgs = img_cut(img, 400, 400, 200)
    img_dir = 'data/helc3_images'
    if os.path.isdir(img_dir) is False:
        os.makedirs(img_dir)
    for index, i in enumerate(imgs):
        cv2.imwrite(os.path.join(img_dir, '%.4d.jpg' % index), i)


def main(args):
    """

    :param args:
    :return:
    """
    img = cv2.imread(args.image)
    imgs = img_cut(img, args.width, args.height, args.step)
    img_dir = args.output
    for index, i in enumerate(imgs):
        cv2.imwrite(os.path.join(img_dir, '%.4d.jpg' % index), i)
    print('split image %s into %s for %d slice'%(args.image, args.output, len(imgs)))


