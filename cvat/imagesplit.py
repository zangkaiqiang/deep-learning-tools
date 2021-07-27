"""
图像文件的批量切割
1. 支持单个图像文件的切割
2. 支持文件夹中所有图像的切割
"""
import os
import cv2
import argparse


def img_cut(img_zero, width, height, step):
    """

    :param img_zero:
    :param width:
    :param height:
    :param step:
    :return:
    """
    imgs = []
    # step = 200
    # width = 400
    # height = 400

    for i in range(1000):
        left = i * step
        right = left + width
        if right > img_zero.shape[1]:
            break

        for j in range(1000):
            top = j * step
            bottom = top + height
            if bottom > img_zero.shape[0]:
                break
            imgs.append(img_zero[top:top + height, left:left + width])

    return imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits images to small image.')
    parser.add_argument('image', metavar='cut_image', type=str, help='Path to cut the image')
    parser.add_argument('output', metavar='output_dir', type=str)
    parser.add_argument('-w', dest='width', type=int)
    parser.add_argument('-t', dest='height', type=int)
    parser.add_argument('-s', dest='step', type=int)
    parser.add_argument('-p', dest='prefix', type=str, default='')
    parser.add_argument('-suf', dest='suffix', type=str, default='')
    args = parser.parse_args()

    if os.path.isdir(args.output) is False:
        os.makedirs(args.output)

    imgs = []
    if os.path.isdir(args.image):
        images_name = os.listdir(args.image)
        images_name.sort()
        images_path = [os.path.join(args.image, name) for name in images_name]

        for image_path in images_path:
            img = cv2.imread(image_path)
            imgs.extend(img_cut(img, args.width, args.height, args.step))
    else:
        img = cv2.imread(args.image)
        imgs = img_cut(img, args.width, args.height, args.step)

    img_dir = args.output
    if args.prefix != '':
        args.prefix += '_'
    if args.suffix != '':
        args.suffix = '_'+args.suffix
    for index, i in enumerate(imgs):
        cv2.imwrite(os.path.join(img_dir, '%s%.4d%s.jpg' % (args.prefix, index, args.suffix)), i)

    print('split image %s into %s for %d slice' % (args.image, args.output, len(imgs)))
