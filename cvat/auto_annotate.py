"""

"""
import os
import numpy as np
import cv2
from bs4 import BeautifulSoup
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from cvat_annotation_split import cnts2tag
from train_config import *


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")# path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

origin_image_dir = 'annotations/HE_TASK_SPLIT/images'
annotations_path = 'annotations/HE_TASK_SPLIT/annotations.xml'
output_annotations = 'output/xxxx.xml'


def generate_cnts():
    """
    返回字典
    key：image name
    value：image cnts
    :return:
    """
    # 加载训练好的模型

    # 加载图像
    images_name = os.listdir(origin_image_dir)
    images_path = [os.path.join(origin_image_dir, i) for i in images_name]
    annotation = BeautifulSoup(open(annotations_path))
    labels_tag = annotation.find_all('label')
    # 生成字典
    cnts_dict = {}
    labels_dict = {}
    for image_path, image_name in zip(images_path, images_name):
        img = cv2.imread(image_path)
        p = predictor(img)
        if len(p['instances']) == 0:
            continue

        contours = []
        labels = []
        for mask_cuda, label_cuda in zip(p['instances'].pred_masks, p['instances'].pred_classes):
            label = np.int(label_cuda.cpu())
            labels.append(labels_tag[label].find('name').text)
            mask = mask_cuda.cpu()
            mask = np.array(mask).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours.append(cnts[0])

        cnts_dict[image_name] = contours
        labels_dict[image_name] = labels
    return cnts_dict, labels_dict


def rebuild_annotations():
    """

    :return:
    """
    # 加载原有注释
    anno = BeautifulSoup(open(annotations_path))
    image_tags = anno.find_all('image')
    # 加载cnts, labels
    cnts_dict, labels_dict = generate_cnts()

    # append cnts to tags
    for tag in image_tags:
        image_name = tag['name']
        if image_name not in cnts_dict.keys():
            continue
        image_cnts = cnts_dict[image_name]
        cnts_label = labels_dict[image_name]
        cnts2tag(tag, image_cnts, cnts_label)

    # export labels xml
    with open(output_annotations, 'w') as f:
        f.write(anno.prettify())


if __name__ == '__main__':
    rebuild_annotations()
