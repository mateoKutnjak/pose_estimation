import os
import json
import scipy.misc
import numpy as np
import random
import cv2
import data_process


class MPII_dataset:

    def __init__(self, images_dir, annots_json_filename):
        self.annots_json_filename = annots_json_filename
        self.images_dir = images_dir
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

        self.create_dataset()

    def create_dataset(self):
        with open(self.annots_json_filename) as f:
            json_parsed = json.loads(f.read())

        for index, value in enumerate(json_parsed):
            self.annots_valid.append(value) if 'isValidation' in value else self.annots_train.append(value)

    def create_batches(self):
        pass

    def process_image(self, annotation, flip_flag, scale_flag, rotation_flag):
        image_filename = annotation['img_paths']
        image = scipy.misc.imread(os.path.join(self.images_dir, image_filename))

        obj_center = np.array(annotation['objpos'])
        obj_joints = np.array(annotation['joint_self'])
        scale = annotation['scale_provided']

        obj_center[1] += 15 * scale
        scale *= 1.25

        if flip_flag and random.choice([0, 1]):
            image, obj_center, obj_joints = self.flip(image, obj_center, obj_joints)
        if scale_flag:
            scale *= np.random.uniform(0.8, 1.2)
        rotation_angle = np.random.randint(-30, 30) if rotation_flag and random.choice([0, 1]) else 0




    def flip(self, image, center, joints):
        flipped_joints = np.copy(joints)

        im_height, im_width, im_channels = image.shape

        flipped_image = cv2.flip(image, flipCode=1) # COMMENT mirrors image x coordinates
        flipped_joints[:, 0] = im_width - joints[:, 0] # COMMENT mirrors joints x coordinates

        for joint_pair in self.joint_pairs:
            temp = joints[joint_pair[0], :]
            flipped_joints[joint_pair[0], :] = flipped_joints[joint_pair[1], :]
            flipped_joints[joint_pair[1], :] = temp

        flipped_center = center
        flipped_center[0] = im_width - center[0]

        return flipped_image, flipped_center, flipped_joints

