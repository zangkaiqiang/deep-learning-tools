"""
合并两个task的annotation
合并逻辑： 根据imagename以及labelname
需要在三个task，其中有两个是需要合并的task，另一个是合并后的task
新建第三个task时需要将task1和task2中的图像全部放到task3中，然后从task3得到标注模板
"""
from bs4 import BeautifulSoup
import argparse


def anno2dict(anno1_images):
    """
    转换成字典的方式
    key是image_name
    :param anno1_images:
    :return:
    """
    anno1_images_dict = {}
    for anno_image in anno1_images:
        image_name = anno_image['name']
        anno1_images_dict[image_name] = anno_image
    return anno1_images_dict


def anno_merge(a1_path, a2_path, a3_path):
    """

    :param a1_path:
    :param a2_path:
    :param a3_path:
    :return:
    """

    anno1 = BeautifulSoup(open(a1_path), features='lxml')
    anno2 = BeautifulSoup(open(a2_path), features='lxml')

    anno1_images = anno1.find_all('image')
    anno2_images = anno2.find_all('image')

    anno1_images_dict = anno2dict(anno1_images)
    anno2_images_dict = anno2dict(anno2_images)

    # 目标标注文件
    anno = BeautifulSoup(open(a3_path), features='lxml')
    anno_images = anno.find_all('image')
    for anno_image in anno_images:
        image_name = anno_image['name']
        if image_name in anno1_images_dict.keys():
            polygons = anno1_images_dict[image_name].find_all('polygon')
            for polygon in polygons:
                anno_image.append(polygon)

        if image_name in anno2_images_dict.keys():
            polygons = anno2_images_dict[image_name].find_all('polygon')
            for polygon in polygons:
                anno_image.append(polygon)

    return anno


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits cvat annotations to small anno.')
    parser.add_argument('annotation1', metavar='first_annotation', type=str, )
    parser.add_argument('annotation2', metavar='second_annotation', type=str)
    parser.add_argument('annotation3', metavar='target_annotation', type=str)
    parser.add_argument('output_annotation', metavar='output annotation', type=str)

    args = parser.parse_args()
    annotation = anno_merge(args.annotation1, args.annotation2, args.annotation3)

    with open(args.output_annotation, 'w') as f:
        f.write(annotation.prettify())
