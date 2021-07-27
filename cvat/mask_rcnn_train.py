import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances, load_coco_json

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer

# register_coco_instances("my_dataset", {}, "instances_default.json", "image")
# data= DatasetCatalog.get("my_dataset")
# MetadataCatalog.get("my_dataset").set(thing_classes=['lc', 'overlapping', 'impurity', 'roundness', 'gray','unknown'])

coco_instance_path = 'annotations/task_helc3-1-2021_04_23_02_05_18-coco 1.0/annotations/instances_default.json'
register_coco_instances("my_dataset", {}, coco_instance_path, "data/helc3_images")
data = DatasetCatalog.get("my_dataset")
MetadataCatalog.get("my_dataset").set(thing_classes=['lc'])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
# cfg.MODEL.RPN.IOU_THRESHOLDS = [0.2, 0.8]
# cfg.MODEL.RPN.POSITIVE_FRACTION = 0.8
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
