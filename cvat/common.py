"""

"""
import os
import re
import cv2
import numpy as np
from bs4 import BeautifulSoup


def cnts2tag(tag, cnts, labels):
    """
    将轮廓信息添加到tag
    :param tag: 输入的tag
    :param cnts: 轮廓信息
    :param label: 需要标注的标签
    :return:
    """
    for cnt, label in zip(cnts, labels):
        cnt = cv2.approxPolyDP(cnt, 1, True)
        if len(cnt) < 3:
            continue
        cnt = cnt.reshape((-1, 2))
        cnt_points = ';'.join([','.join(map(str, p)) for p in cnt])
        polygon = BeautifulSoup('bs').new_tag('polygon')
        polygon['label'] = label

        polygon['occluded'] = '0'
        polygon['points'] = cnt_points
        tag.append(polygon)
    return tag


def tag2cnts(image_tag):
    """
    poly_tag to cnt
    :param polygon_tag:
    :return:
    """
    cnts = []
    labels = []
    for polygon_tag in image_tag.find_all('polygon'):
        cnt = re.split('[;,]', polygon_tag['points'])
        cnt = np.array([float(i) for i in cnt])
        cnt = cnt.reshape((-1, 1, 2)).astype(int)
        cnts.append(cnt)
        labels.append(polygon_tag['label'])

    return cnts, labels

def mkdirs(dirs):
    """

    :param dirs:
    :return:
    """
    if os.path.isdir(dirs) is False:
        os.makedirs(dirs)