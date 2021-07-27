import os

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data import build_detection_train_loader, DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T

from train_config import *


class Trainer(DefaultTrainer):
    mapper = None

    def __init__(self, cfg, mapper):
        DefaultTrainer.__init__(self, cfg)
        self.mapper = mapper

    @classmethod
    def build_train_loader(self, cfg):
        return build_detection_train_loader(cfg, mapper=self.mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


if __name__ == '__main__':
    mapper = DatasetMapper(cfg, is_train=True, augmentations=[
        # T.Resize((512,512)),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomRotation(angle=[0, 90, 180, 270], sample_style='choice')
    ])

    trainer = Trainer(cfg, mapper)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    # predictor = DefaultPredictor(cfg)
    #
    # evaluator = COCOEvaluator("my_dataset_test", ('bbox',), False, output_dir="./output/")
    # val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    # inference_on_dataset(predictor, val_loader, evaluator)
    # print(trainer.model)

    # evaluator = COCOEvaluator("my_dataset_test", ("bbox", "segm"), False, output_dir="./output/")
    # val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    # print(inference_on_dataset(trainer.model, val_loader, evaluator))
