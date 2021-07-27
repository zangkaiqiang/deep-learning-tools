#!/bin/bash
set -e
for i in {0..4};
do
    echo $i;
    python tools/train.py configs/cell/detectors_cascade_rcnn_r50_1x_coco_cell${i}.py;
done