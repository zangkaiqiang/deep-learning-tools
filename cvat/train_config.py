from detectron2.data.datasets import register_coco_instances

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from common import mkdirs
import time

# HE EOS
# coco_train = 'annotations/HE_EOS_SPLIT/coco/train.json'
# coco_test = 'annotations/HE_EOS_SPLIT/coco/test.json'
# register_coco_instances("my_dataset_train", {}, coco_train, "output/eos")
# register_coco_instances("my_dataset_test", {}, coco_test, "output/eos")

# HELC3
# coco_train = 'annotations/HELC3_SPLIT/coco/instances_default_s100.json'
# coco_test = 'annotations/HELC3_SPLIT/coco/instances_default_s100.json'
# register_coco_instances("my_dataset_train", {}, coco_train, "data/HELC3_SPLIT/s100")
# register_coco_instances("my_dataset_test", {}, coco_test, "data/HELC3_SPLIT/s100")

# register_coco_instances("my_dataset_train", {}, coco_instance_path, "data/train_test_split/train")
# register_coco_instances("my_dataset_test", {}, coco_instance_path, "data/train_test_split/test")
# MetadataCatalog.get("my_dataset_train").set(thing_classes=['lc'])
# MetadataCatalog.get("my_dataset_test").set(thing_classes=['lc'])

# HE 82
# coco_train = 'annotations/HE_082_SPLIT/coco/instances_default.json'
# coco_test = 'annotations/HE_082_SPLIT/coco/instances_default.json'
#
# image_path = 'data/HE_082_S100'
# register_coco_instances("my_dataset_train", {}, coco_train, image_path)
# register_coco_instances("my_dataset_test", {}, coco_test, image_path)

# HE MERGE
# coco_train = 'annotations/HE_MERGE/coco/instances_default.json'
# coco_test = 'annotations/HE_MERGE/coco/instances_default.json'
# image_path = 'data/HE_MERGE'


# HE MERGE2
# coco_train = 'annotations/HE_MERGE2/coco/annotations/instances_default.json'
# coco_test = 'annotations/HE_MERGE2/coco/annotations/instances_default.json'
# image_path = 'annotations/HE_MERGE2/coco/images'

# HE WORK
coco_train = 'annotations/HE_WORK/coco/train.json'
coco_test = 'annotations/HE_WORK/coco/test.json'
image_path = 'annotations/HE_WORK/image_split'

register_coco_instances("my_dataset_train", {}, coco_train, image_path)
register_coco_instances("my_dataset_test", {}, coco_test, image_path)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
# cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 200000
cfg.SOLVER.STEPS = []  # do not decay learning rate
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
cfg.MODEL.DEVICE = "cuda:1"

# cfg.OUTPUT_DIR = "output/HE_MERGE_R101"
# cfg.OUTPUT_DIR = "output/HE_MERGE_%d"%(int(time.time()))
cfg.OUTPUT_DIR = "output/HE_WORK0702"

mkdirs(cfg.OUTPUT_DIR)

cfg.TEST.EVAL_PERIOD = 3000
