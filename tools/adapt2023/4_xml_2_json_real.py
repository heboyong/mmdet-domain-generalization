import json
import xml.etree.ElementTree as ET
import os
import numpy as np


def load_load_image_labels(LABEL_PATH, class_name=[]):
    # temp=[]
    images = []
    type = "instances"
    annotations = []
    # assign your categories which contain the classname and calss id
    # the order must be same as the class_nmae
    new_classes = ["Top Casing", "Gear", "Bottom Casing", "Carrier", "Cover", "Housing", "Planet", "Sun", "Button", "Cam",
               "Case End", "Case Middle"]
    categories = [{"name": "1", "id": 1, "supercategory": "none"}, {"name": "2", "id": 2, "supercategory": "none"},
                  {"name": "3", "id": 3, "supercategory": "none"}, {"name": "4", "id": 4, "supercategory": "none"},
                  {"name": "5", "id": 5, "supercategory": "none"}, {"name": "6", "id": 6, "supercategory": "none"},
                  {"name": "7", "id": 7, "supercategory": "none"}, {"name": "8", "id": 8, "supercategory": "none"},
                  {"name": "9", "id": 9, "supercategory": "none"}, {"name": "10", "id": 10, "supercategory": "none"},
                  {"name": "11", "id": 11, "supercategory": "none"}, {"name": "12", "id": 12, "supercategory": "none"}]
    # load ground-truth from xml annotations
    id_number = 0
    for image_id, label_file_name in enumerate(os.listdir(LABEL_PATH)):

        label_file = LABEL_PATH + label_file_name
        image_file = label_file_name.split('.')[0] + '.png'
        print(image_file)
        tree = ET.parse(label_file)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        images.append({
            "file_name": image_file,
            "height": height,
            "width": width,
            "id": str(image_file).split('.')[0]
        })  # id of the image. referenced in the annotation "image_id"

        for anno_id, obj in enumerate(root.iter('object')):
            name = obj.find('name').text
            bbox = obj.find('bndbox')

            cls_id = new_classes.index(name) + 1
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xlen = xmax - xmin
            ylen = ymax - ymin
            annotations.append({
                "segmentation": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin],
                "area": xlen * ylen,
                "iscrowd": 0,
                "image_id": str(image_file).split('.')[0],
                "bbox": [xmin, ymin, xlen, ylen],
                "category_id": cls_id,
                "id": id_number,
                "ignore": 0
            })
            # print([image_file,image_id, cls_id, xmin, ymin, xlen, ylen])
            id_number += 1

    return {"images": images, "annotations": annotations, "categories": categories}


if __name__ == '__main__':
    LABEL_PATH = '/home/kemove/disk/data/det/adapt2023/real/xmls/'
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    label_dict = load_load_image_labels(LABEL_PATH, classes)

    with open('real_test.json', 'w') as json_file:
        json_file.write(json.dumps(label_dict, ensure_ascii=False))
        json_file.close()
