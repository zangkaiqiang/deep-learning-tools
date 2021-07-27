import pytest
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from cell_recogination.recogination import sample_fill_contours, sample_draw_contours
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from train_config import *

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.TEST.DETECTIONS_PER_IMAGE = 200
predictor = DefaultPredictor(cfg)

# data = ['image/000.jpg',
#         'image/001.jpg',
#         'image/002.jpg']

test_image_dir = 'annotations/HE_WORK/image_split_test'
# test_image_path = [[os.path.join(test_image_dir, name, '%s.jpg' % name) for name in os.listdir(test_image_dir)]]
# test_image_path.sort()

# data = [os.path.join('output/cut_result', i) for i in os.listdir('output/cut_result')]
data = [os.path.join(test_image_dir, i) for i in os.listdir(test_image_dir)]
data.sort()

images_dir = '/share/pathology/肺泡融合扫描图-20201228'
images_path = [os.path.join(images_dir, name, '%s.BMAP' % name) for name in os.listdir(images_dir)]
images_path.sort()
cell_color = [67, 34, 89]

nt = int(time.time())


@pytest.fixture(scope='session', params=images_path)
def bmap_file(request):
    return request.param


# @pytest.fixture(scope='session')
# def predictorx():
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     # cfg.DATASETS.TRAIN = ("my_dataset",)
#     # cfg.DATASETS.TEST = ()
#     # cfg.DATALOADER.NUM_WORKERS = 2
#     # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#     cfg.SOLVER.MAX_ITER = 1000
#     cfg.SOLVER.STEPS = []  # do not decay learning rate
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
#
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
#     predictor = DefaultPredictor(cfg)
#     return predictor


@pytest.fixture(scope='session', params=data)
def paramsx(request):
    return request.param


def test_model(paramsx):
    im = cv2.imread(paramsx)
    p = predictor(im)

    # contours = []
    # for mask_cuda in p['instances'].pred_masks:
    #     mask = mask_cuda.cpu()
    #     mask = np.array(mask).astype(np.uint8)
    #     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     contours.append(cnts[0])
    metadatacatalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=['eosinophil',
                                                                                    'neutrophil',
                                                                                    'monocytes',
                                                                                    'lymphocyte',
                                                                                    'macrophage',
                                                                                    'hematocyte',
                                                                                    'Basophil',
                                                                                    'alveolar1',
                                                                                    'alveolar2',
                                                                                    'fibroblast',
                                                                                    'tracheal_epithelial_cell',
                                                                                    'unknown2'])
    v = Visualizer(im[:, :, ::-1], metadatacatalog, scale=1.2)
    out = v.draw_instance_predictions(p["instances"].to("cpu"))
    out_image = out.get_image()[:,:,::-1]

    # plt.figure(figsize=(10, 10))
    # plt.imshow(sample_draw_contours(contours, img_test))
    # plt.show()
    img_output_dir = os.path.join(cfg.OUTPUT_DIR, 'image')
    mkdirs(img_output_dir)
    img_output_path = os.path.join(img_output_dir, '%s.jpg' % paramsx.split('/')[-1][:-4])
    cv2.imwrite(img_output_path, out_image)


def test_evaluation():
    """

    :return:
    """


def test_run(predictorx, region_roix, bmap_file):
    """

    :param predictorx:
    :param paramsx:
    :return:
    """
    file_name = bmap_file.split('/')[-1].split('.')[0]
    img_rois = region_roix[0]
    positions = region_roix[1]
    positions1x = positions / 40
    cell_nums = run(img_rois, predictorx)
    img1x = read_mat_small(bmap_file, 1)
    for p, n in zip(positions1x, cell_nums):
        cv2.putText(img1x, str(n), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
    plt.imshow(img1x)
    plt.show()
    cv2.imwrite('output/%d_%s.jpg' % (nt, file_name), img1x)

    print(cell_nums)
    assert True


def test_predict(predictorx, image_file):
    """

    :param predictorx:
    :param image_file:
    :return:
    """
    img_test = cv2.imread(image_file)
    p = predictorx(img_test)

    contours = []
    for mask_cuda in p['instances'].pred_masks:
        mask = mask_cuda.cpu()
        mask = np.array(mask).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.append(cnts[0])


@pytest.fixture(scope='session')
def region_roix(bmap_file):
    img_rois, positions = region_roi(bmap_file, cell_color)
    return img_rois, positions


def test_region_roi():
    """

    :return:
    """
