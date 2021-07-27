"""
cross validation script
"""
config_path = 'configs/cell/detectors_cascade_rcnn_r50_1x_coco_cell.py'

c = open(config_path, 'r').read()
for i in range(5):
    s = """
data = dict(
train=dict(
    img_prefix='annotations/HE_WORK_412x412/image_split_412x412',
    classes=classes,
    ann_file='annotations/HE_WORK_412x412/coco/train{}.json'),
val=dict(
    img_prefix='annotations/HE_WORK_412x412/image_split_412x412',
    classes=classes,
    ann_file='annotations/HE_WORK_412x412/coco/test{}.json'),
test=dict(
    img_prefix='annotations/HE_WORK_412x412/image_split_412x412',
    classes=classes,
    ann_file='annotations/HE_WORK_412x412/coco/test{}.json'))
    """.format(i, i, i)

    with open('{}{}.py'.format(config_path.split('.')[0], i), 'w') as f:
        f.write(c + ('\n') + s)
        f.close()
