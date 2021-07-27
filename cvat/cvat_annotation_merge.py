"""
合并标注
support multi-image to multi-image
"""
from common import cnts2tag, tag2cnts
from bs4 import BeautifulSoup
import argparse


def cnts_merge(shape, cnts, labels, width, height, step):
    """
    和split
    merge cnts
    return index
    :param shape:
    :param cnts: cnts_dict
    :param labels:
    :param width:
    :param height:
    :param step:
    :return:
    """
    # store result
    result_cnts = []
    result_labels = []
    index = 0
    for k in range(1000):
        # store cnts and labels of one big image
        cnts_merged = []
        labels_merged = []
        # 确定边界
        for i in range(1000):
            left = i * step
            right = left + width

            # out of edge
            if right > shape[1]:
                break
            for j in range(1000):
                top = j * step
                bottom = top + height

                # judge out of edge
                if bottom > shape[0]:
                    break

                cnts_merged.extend([cnt + [left, top] for cnt in cnts[index]])
                labels_merged.extend(labels[index])
                index = index + 1

        result_cnts.append(cnts_merged)
        result_labels.append(labels_merged)
        if index >= len(cnts):
            break

    return result_cnts, result_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge cvat annotations to one')
    parser.add_argument('annotation_split', metavar='annotation_split', type=str, help='Path to source annotation')
    parser.add_argument('annotation_merge', metavar='annotation_merge', type=str)
    parser.add_argument('output_annotation', metavar='output annotation', type=str)
    parser.add_argument('-w', dest='width', type=int)
    parser.add_argument('-t', dest='height', type=int)
    parser.add_argument('-s', dest='step', type=int)
    args = parser.parse_args()

    splited_annotations = BeautifulSoup(open(args.annotation_split), features="lxml")
    merged_annotations = BeautifulSoup(open(args.annotation_merge), features="lxml")

    width = args.width
    height = args.height
    step = args.step

    # all images is the same size
    merged_image_tags = merged_annotations.find_all('image')
    image_shape = (int(merged_image_tags[0]['height']), int(merged_image_tags[0]['width']))

    # get image tags from splited annotations
    image_tags = splited_annotations.find_all('image')
    image_cnts_dict = {}
    image_label_dict = {}
    for image_tag in image_tags:
        image_cnts, image_labels = tag2cnts(image_tag)
        image_cnts_dict[int(image_tag['id'])] = image_cnts
        image_label_dict[int(image_tag['id'])] = image_labels

    # support multi origin images
    all_cnts, all_labels = cnts_merge(image_shape, image_cnts_dict, image_label_dict, width, height, step)

    for merged_image_tag in merged_image_tags:
        merged_image_tag.clear()
        id = int(merged_image_tag['id'])
        cnts = all_cnts[id]
        labels = all_labels[id]
        cnts2tag(merged_image_tag, cnts, labels)

    with open(args.output_annotation, 'w') as f:
        f.write(merged_annotations.prettify())
