# cvat系统的标注数据处理
## 数据的处理
1. 标注
2. 使用maskrcnn训练得到模型
3. 使用模型预测新的数据
4. 开发接口将预测结果加载到cvat
5. 人工修正预测结果
6. 将修正的预测结果作为训练数据迭代

## 模型的建立以及训练优化


# 病理图像分析
1. 训练预测模型
2. ROI提取
3. ROI的切割
4. ROI的预测

# 测试

# cocosplit run
```bash
python3 cocosplit.py  --having-annotations -s 0.8 --shuffle \
annotations/HE_WORK/coco/instances_default.json \
annotations/HE_WORK/coco/train.json \
annotations/HE_WORK/coco/test.json

python3 cocosplit.py  --having-annotations -s 0.8 \
annotations/HE_WORK/coco/instances_default.json \
annotations/HE_WORK/coco/train.json \
annotations/HE_WORK/coco/test.json

python3 cocosplit.py  --having-annotations --kfold \
annotations/HE_WORK_412x412/coco/instances_default.json \
annotations/HE_WORK_412x412/coco/train \
annotations/HE_WORK_412x412/coco/test
```

# imagesplit
## sigle image
```bash
python3 imagesplit.py -w 400 -t 400 -s 200 data/1010/13-COPD-F-003-1.png output/xxxx -p prefix
```
## for image dir
```bash
python3 imagesplit.py -w 400 -t 400 -s 400 annotations/HE_TASK/images output/HE_TASK_SPLIT -p prefix

python3 imagesplit.py -w 512 -t 512 -s 128 annotations/HE_WORK/image annotations/HE_WORK/image_split -p he_work
```


# cvat_annotation_split
```bash
# support multi-image annotation
python3 cvat_annotation_split.py -w 400 -t 400 -s 200 annotations/HELC3/annotations.xml annotations/HELC3_SPLIT/annotations.xml output/split_xx.xml

python3 cvat_annotation_split.py -w 512 -t 512 -s 128 annotations/HE_TASK/annotations.xml annotations/HE_WORK/annotations.xml output/split_xx.xml
```

# cvat_annotation_merge
```bash
# support multi-split-image to multi-origin-image
python3 cvat_annotation_merge.py -w 400 -t 400 -s 200 annotations/HELC3_SPLIT/annotations.xml annotations/HELC3/annotations.xml output/merge_xx.xml

# multi to multi
python3 cvat_annotation_merge.py -w 400 -t 400 -s 400 annotations/HE_TASK_SPLIT/annotations.xml annotations/HE_TASK/annotations.xml output/multi-merge_xx.xml
```

# cvat task merge
```shell
python3 cvat_tasks_merge.py annotations/HELC3_SPLIT/s100/annotations.xml annotations/HE_082_SPLIT/s100/annotations.xml annotations/HE_MERGE/annotations.xml output/task_merge.xml
```

# annotation_cut
```shell
python3 annotations_cut.py annotations/HE_TASK/annotations.xml -id 18
```

# mmdet for cell
```shell
# mask_rcnn
# train
python tools/train.py configs/cell/mask_rcnn_r50_fpn_poly_1x_coco_cell.py
# test
python tools/test.py configs/cell/mask_rcnn_r50_fpn_poly_1x_coco_cell.py work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/latest.pth --eval bbox segm

# cascade_rcnn
# train
python tools/train.py configs/cell/cascade_rcnn_r50_fpn_1x_coco_cell.py
# test
python tools/test.py configs/cell/cascade_rcnn_r50_fpn_1x_coco_cell.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_cell/latest.pth --eval bbox

# retinanet
# train
python tools/train.py configs/cell/retinanet_r50_fpn_1x_coco_cell.py
# test
python tools/test.py configs/cell/retinanet_r50_fpn_1x_coco_cell.py work_dirs/retinanet_r50_fpn_1x_coco_cell/latest.pth --eval bbox


# detectoRS
# train
python tools/train.py configs/cell/detectors_cascade_rcnn_r50_1x_coco_cell.py
# test
python tools/test.py configs/cell/detectors_cascade_rcnn_r50_1x_coco_cell.py work_dirs/detectors_cascade_rcnn_r50_1x_coco_cell/latest.pth --eval bbox

# detectoRS HTC
python tools/train.py configs/cell/detectors_htc_rcnn_r50_1x_coco_cell.py

# compare 
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]

python tools/analysis_tools/analyze_logs.py plot_curve \
work_dirs/detectors_cascade_rcnn_r50_1x_coco_cell/20210706_090754.log.json \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210705_171846.log.json \
work_dirs/cascade_rcnn_r50_fpn_1x_coco_cell/20210706_100346.log.json \
work_dirs/retinanet_r50_fpn_1x_coco_cell/20210706_101736.log.json \
--keys bbox_mAP_50 --title performance_bbox_map_50 --legend detectors mask_rcnn cascade_rcnn retinanet \
--out output/ap50.jpg

python tools/analysis_tools/analyze_logs.py plot_curve \
work_dirs/detectors_cascade_rcnn_r50_1x_coco_cell/20210706_090754.log.json \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210705_171846.log.json \
work_dirs/cascade_rcnn_r50_fpn_1x_coco_cell/20210706_100346.log.json \
work_dirs/retinanet_r50_fpn_1x_coco_cell/20210706_101736.log.json \
--keys bbox_mAP --title performance_bbox_map --legend detectors mask_rcnn cascade_rcnn retinanet \
--out output/ap.jpg

python tools/analysis_tools/analyze_logs.py plot_curve \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210707_093612.log.json \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210707_100619.log.json \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210707_103308.log.json \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210705_171846.log.json \
work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/20210707_105645.log.json \
--keys bbox_mAP_50 --legend 0.5 0.6 0.7 0.8 0.8-2 --out output/train_test_result.jpg


# analysis per category
python tools/test.py \
       configs/cell/mask_rcnn_r50_fpn_poly_1x_coco_cell.py \
       work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/latest.pth \
       --format-only \
       --options "jsonfile_prefix=./results/mask_rcnn"

python tools/analysis_tools/coco_error_analysis.py \
       results/results.segm.json \
       results/mask_rcnn_results \
       --ann=annotations/HE_WORK_412x412/coco/test.json \
       --types='segm'
       
python tools/analysis_tools/coco_error_analysis.py \
       results/results.bbox.json \
       results/mask_rcnn_results \
       --ann=annotations/HE_WORK_412x412/coco/test.json \
       --types='bbox'

```
# FrameWork
![frame](./images/HE染色肺部切片影像识别.png)
# Performance
## compare detectoRS mask_rcnn  cascade_rcnn retinanet
![ap50](./images/ap50.jpg)
![ap](./images/ap.jpg)

## k-fold for k = 5 for mask_rcnn
![cv5](./images/train_test_cv5.jpg)

## Performance analysis for per category
![bbox-allclass](./images/mask_rcnn_results/bbox/bbox-allclass-allarea.png)
![bbox-eosinophil-allarea.png](./images/mask_rcnn_results/bbox/bbox-eosinophil-allarea.png)
![bbox-lymphocyte-allarea.png](./images/mask_rcnn_results/bbox/bbox-lymphocyte-allarea.png)
![bbox-macrophage-allarea.png](./images/mask_rcnn_results/bbox/bbox-macrophage-allarea.png)
![bbox-monocytes-allarea.png](./images/mask_rcnn_results/bbox/bbox-monocytes-allarea.png)
![bbox-alveolar1-allarea.png](./images/mask_rcnn_results/bbox/bbox-alveolar1-allarea.png)
![bbox-alveolar2-allarea.png](./images/mask_rcnn_results/bbox/bbox-alveolar2-allarea.png)
![bbox-tracheal_epithelial_cell-allarea.png](./images/mask_rcnn_results/bbox/bbox-tracheal_epithelial_cell-allarea.png)
![bbox-neutrophil-allarea.png](./images/mask_rcnn_results/bbox/bbox-neutrophil-allarea.png)
![bbox-Basophil-allarea.png](./images/mask_rcnn_results/bbox/bbox-Basophil-allarea.png)
![bbox-fibroblast-allarea.png](./images/mask_rcnn_results/bbox/bbox-fibroblast-allarea.png)
![bbox-hematocyte-allarea.png](./images/mask_rcnn_results/bbox/bbox-hematocyte-allarea.png)

# DetVisGui
```shell
# Display the COCO bounding box groundtruth
python DetVisGUI.py configs/cell/mask_rcnn_r50_fpn_poly_1x_coco_cell.py

# Display the validation results of COCO detection by json output file
python DetVisGUI.py configs/cell/mask_rcnn_r50_fpn_poly_1x_coco_cell.py --det_file results/mask_rcnn.segm.json

# Display the mask rcnn results
python DetVisGUI_test.py configs/cell/mask_rcnn_r50_fpn_poly_1x_coco_cell.py work_dirs/mask_rcnn_r50_fpn_poly_1x_coco_cell/latest.pth annotations/HE_WORK_412x412/image_split_412x412 --device cpu
```
