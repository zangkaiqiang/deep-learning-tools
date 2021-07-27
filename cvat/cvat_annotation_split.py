"""
标注数据的生成
"""
from bs4 import BeautifulSoup
import numpy as np
import re
from common import cnts2tag
from cell_recogination.common import get_sample_area_info
import argparse


def cnt_split(shape, cnts, labels, width, height, step):
    """
    和image cut 配套
    return index
    :param shape:
    :param cnts:
    :param width:
    :param height:
    :param step:
    :return:
    """
    result_labels = []
    result_cnts = []
    # 确定边界
    for i in range(1000):
        left = i * step
        right = left + width
        if right > shape[1]:
            break
        for j in range(1000):
            top = j * step
            bottom = top + height
            if bottom > shape[0]:
                break
            # FILTER
            move_cnts = [cnt - [left, top] for cnt in cnts]
            df_move_info = get_sample_area_info(move_cnts)
            df_move_info['labels'] = labels
            df_move_info = df_move_info[(df_move_info.x >= 0) & (df_move_info.x < width)]
            df_move_info = df_move_info[(df_move_info.y >= 0) & (df_move_info.y < height)]
            move_cnts = [move_cnts[i] for i in df_move_info.index]
            result_cnts.append(move_cnts)
            result_labels.append(list(df_move_info['labels']))

    return result_labels, result_cnts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits cvat annotations to small anno.')
    parser.add_argument('annotation', metavar='source_annotation', type=str, help='Path to source annotation')
    parser.add_argument('annotation_split', metavar='annotation_split', type=str)
    parser.add_argument('output_annotation', metavar='output annotation', type=str)
    parser.add_argument('-w', dest='width', type=int)
    parser.add_argument('-t', dest='height', type=int)
    parser.add_argument('-s', dest='step', type=int)
    args = parser.parse_args()

    source_annotation = BeautifulSoup(open(args.annotation))
    image_tags = source_annotation.find_all('image')
    image_tag_dict = {}
    for image_tag in image_tags:
        image_name = image_tag['name']
        image_tag_dict[image_name] = image_tag
    images_name = list(image_tag_dict.keys())
    images_name.sort()

    result_labels = []
    result_cnts = []
    for image_name in images_name:
        image_tag = image_tag_dict[image_name]

        source_cnts = []
        source_labels = []
        image_shape = (int(image_tag['height']), int(image_tag['width']))

        source_polygon_tags = image_tag.find_all('polygon')

        for polygon_tag in source_polygon_tags:
            source_labels.append(polygon_tag['label'])
            cnt = re.split('[;,]', polygon_tag['points'])
            cnt = np.array([float(i) for i in cnt])
            cnt = cnt.reshape((-1, 1, 2)).astype(int)
            source_cnts.append(cnt)

        cnts_splited_labels, cnts_splited = cnt_split(image_shape, source_cnts, source_labels, args.width, args.height,
                                                      args.step)
        result_labels.extend(cnts_splited_labels)
        result_cnts.extend(cnts_splited)

    split_annotation = BeautifulSoup(open(args.annotation_split))
    split_annotation_tags = split_annotation.find_all('image')
    for tag in split_annotation_tags:
        tag_id = int(tag['id'])
        if tag_id >= len(result_cnts):
            continue
        cnts = result_cnts[tag_id]
        labels = result_labels[tag_id]
        cnts2tag(tag, cnts, labels)

    with open(args.output_annotation, 'w') as f:
        f.write(split_annotation.prettify())
