import os
import json
import scipy.misc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as rot
import preprocessing


class MPII_dataset:

    def __init__(self, images_dir, annots_json_filename, input_res, output_res):
        self.annots_json_filename = annots_json_filename
        self.images_dir = images_dir
        self.input_res = input_res
        self.output_res = output_res

        self.annots_train = []
        self.annots_valid = []

        self.joints_num = 16
        self.joint_pairs = (
            [0, 5],     # COMMENT ankles
            [1, 4],     # COMMENT knees
            [2, 3],     # COMMENT hips
            [10, 15],   # COMMENT wrists
            [11, 14],   # COMMENT elbows
            [12, 13]    # COMMENT shoulders
        )

    def create_dataset(self):
        with open(self.annots_json_filename) as f:
            json_parsed = json.loads(f.read())

        for index, value in enumerate(json_parsed):
            self.annots_valid.append(value) if value['isValidation'] == 1.0 else self.annots_train.append(value)

    # def create_batches(self, batch_size):
    #     train_input = np.zeros(shape=(batch_size, self.input_res[0], self.input_res[1], 3))
    #     heatmap_putput = np.zeros(shape=(batch_size, self.output_res[0], self.output_res[1], self.joints_num))
    #
    #     while True:
    #         for index, value in self.an

    def process_image(self, annotation, flip_flag, scale_flag, rotation_flag):
        image_filename = annotation['img_paths']
        image = scipy.misc.imread(os.path.join(self.images_dir, image_filename))

        obj_center = np.array(annotation['objpos'])
        obj_joints = np.array(annotation['joint_self'])
        obj_joint_visibility = obj_joints[:, 2]
        obj_joints = obj_joints[:, :2]
        scale = annotation['scale_provided']

        # COMMENT To avoid joints cropping
        scale *= 1.25
        angle = 0

        if flip_flag and np.random.sample() > 0.5:
            image, obj_center, obj_joints = preprocessing.flip(image, obj_center, obj_joints)
        if scale_flag and np.random.sample() > 0.5:
            scale *= np.random.uniform(0.75, 1.25)
        if rotation_flag and np.random.sample() > 0.5:
            angle = np.random.randint(-30, 30)
            image, obj_center, obj_joints = preprocessing.rotate(image, obj_center, obj_joints, angle)

        preprocessing.draw_processed_image(image, obj_center, obj_joints, scale, angle)

        cropped_image, obj_center, obj_joints = preprocessing.crop(image, obj_center, obj_joints, scale)

        preprocessing.draw_processed_image(cropped_image, obj_center, obj_joints, scale, angle, draw_bbox=False)