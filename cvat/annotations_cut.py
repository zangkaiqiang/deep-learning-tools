"""
标注截断
"""
from bs4 import BeautifulSoup
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cut cvat annotations by image id.')
    parser.add_argument('annotation', metavar='annotation', type=str, help='Path to source annotation')
    parser.add_argument('-id', dest='id', type=int)
    args = parser.parse_args()
    annotation = BeautifulSoup(open(args.annotation), features="lxml")
    image_tags = annotation.find_all('image')
    for image_tag in image_tags:
        if int(image_tag['id']) > args.id:
            image_tag.decompose()
        else:
            print(image_tag['name'])

    with open(args.annotation, 'w') as f:
        f.write(annotation.prettify())
