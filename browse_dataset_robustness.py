import os
import cv2
import json

from imagecorruptions import corrupt

corrupt_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

with open('data/cityscapes/test.json', 'r') as load_f:
    f = json.load(load_f)
image_list = [file['file_name'] for file in f['images']]

image_dict_list = []
for image_name in image_list:
    image_dict = {}
    image = cv2.imread(os.path.join('data/cityscapes/JPEGImages', image_name))
    image_dict['name'] = image_name
    image_dict['image'] = image
    image_dict_list.append(image_dict)

for corrupt_name in corrupt_list:
    for severity in range(1, 6):
        save_path = os.path.join('robustness/', str(corrupt_name) + str(severity))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for ima_dict in image_dict_list:
            name = ima_dict['name']
            image = ima_dict['image']
            print(corrupt_name, severity, name)
            corrupted_image = corrupt(image, corruption_name=corrupt_name, severity=severity)
            cv2.imwrite(os.path.join(save_path, name), corrupted_image)
