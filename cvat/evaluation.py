import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from cell_recogination.recogination import sample_fill_contours, sample_draw_contours
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from train_config import *

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.TEST.DETECTIONS_PER_IMAGE = 200
predictor = DefaultPredictor(cfg)


def evaluate():
    """

    :return:
    """
    evaluator = COCOEvaluator("my_dataset_test", ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

