"""
merge overlapping cells
"""
from bs4 import BeautifulSoup
import cv2
from cell_recogination.recogination import get_sample_area_info
from annotation import cnts2tag

# read
anno = BeautifulSoup(open('data/HELC5-2 /annotations.xml'))
img_seg_class = cv2.imread('data/HELC5/task_helc5-2021_03_22_03_16_46-segmentation mask 1.1/SegmentationClass/roi1.png', 0)
img_seg_class[img_seg_class < 100] = 0
img_seg_class[img_seg_class >= 100] = 255

contours, _ = cv2.findContours(img_seg_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

df_info = get_sample_area_info(contours)
# filter lc and overlapping
lc_index = df_info[df_info.area<=450].index
overlapping_index = df_info[df_info.area>450].index

lc_cnts = [contours[i] for i in lc_index]
overlapping_cnts = [contours[i] for i in overlapping_index]

tags = anno.find_all('image')
tag = tags[0]
tag = cnts2tag(tag, lc_cnts, 'lc')
tag = cnts2tag(tag, overlapping_cnts, 'overlapping')

with open('output/HELC5-2/annotations.xml', 'w') as f:
    f.write(anno.prettify())
