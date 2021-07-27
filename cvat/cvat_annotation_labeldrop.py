"""
drop some label from cvat annotation xml
"""
from bs4 import BeautifulSoup

if __name__ == '__main__':
    annotation_path = 'annotations/HE_TASK/annotations.xml'
    annotation = BeautifulSoup(open(annotation_path),'lxml')
    polygon_tags = annotation.find_all('polygon')
    for tag in polygon_tags:
        if tag['label'] == 'hematocyte':
            tag.decompose()

    with open(annotation_path, 'w') as f:
        f.write(annotation.prettify())
