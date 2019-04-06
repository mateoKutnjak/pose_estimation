import scipy.misc
from scipy.ndimage import rotate as rot
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import cv2
import math


IMAGE_DIR = 'images/'
ANNOTATION_JSON_FILENAME = 'annotations/mpii_annotations.json'

joint_pairs = (
            [0, 5],     # COMMENT ankles
            [1, 4],     # COMMENT knees
            [2, 3],     # COMMENT hips
            [10, 15],   # COMMENT wrists
            [11, 14],   # COMMENT elbows
            [12, 13]    # COMMENT shoulders
        )


def process_image(annotation, flip_flag, scale_flag, rotation_flag):
    image_filename = annotation['img_paths']
    image = scipy.misc.imread(os.path.join(IMAGE_DIR, image_filename))

    obj_center = np.array(annotation['objpos'])
    obj_joints = np.array(annotation['joint_self'])
    scale = annotation['scale_provided']

    # COMMENT To avoid joints cropping
    scale *= 1.25

    # if flip_flag and np.random.sample() > 0.5:
    #     image, obj_center, obj_joints = flip(image, obj_center, obj_joints)
    # if scale_flag and np.random.sample() > 0.5:
    #     scale *= np.random.uniform(0.75, 1.25)
    # if rotation_flag and np.random.sample() > 0.5:
    #     image, obj_center, obj_joints = rotate(image, obj_center, obj_joints)

    x0 = int(obj_center[0] - 200 * scale / 2)
    y0 = int(obj_center[1] - 200 * scale / 2)
    x1 = int(obj_center[0] + 200 * scale / 2)
    y1 = int(obj_center[1] + 200 * scale / 2)

    plt.imshow(image)
    plt.scatter(obj_center[0], obj_center[1])
    plt.plot([x0, x1], [y0, y0])
    plt.plot([x0, x1], [y1, y1])
    plt.plot([x0, x0], [y0, y1])
    plt.plot([x1, x1], [y0, y1])
    plt.show()



    image, obj_center, obj_joints = rotate(image, obj_center, obj_joints, 30)


    x0 = int(obj_center[0] - 200 * scale / 2)
    y0 = int(obj_center[1] - 200 * scale / 2)
    x1 = int(obj_center[0] + 200 * scale / 2)
    y1 = int(obj_center[1] + 200 * scale / 2)

    plt.imshow(image)
    plt.scatter(obj_center[0], obj_center[1])
    plt.plot([x0, x1], [y0, y0])
    plt.plot([x0, x1], [y1, y1])
    plt.plot([x0, x0], [y0, y1])
    plt.plot([x1, x1], [y0, y1])
    plt.show()

    image = crop(image, x0, y0, x1, y1)

    plt.imshow(image)
    plt.scatter(image.shape[0] // 2, image.shape[1] // 2)
    plt.show()

    exit()

    if flip_flag:
        image, obj_center, obj_joints = flip(image, obj_center, obj_joints)

    # COMMENT data augmentation (scaling)
    if scale_flag:
        scale *= np.random.uniform(0.75, 1.25)

    # COMMENT data augmentation (rotation)
    # rotation_angle = np.random.randint(-30, 30) if rotation_flag and np.random.sample() > 0.5 else 0
    rotation_angle = 30

    image = rotate(image, 30)

    draw_with_joints(image, obj_joints)


def rotate(image, obj_center, obj_joints, angle):
    rotated_image = rot(image, angle)
    rotation_origin = (image.shape[0] // 2, image.shape[1] // 2)

    joint_visibility = obj_joints[:, 2]
    obj_joints = obj_joints[:, :2]

    points = [obj_center]
    points.extend(obj_joints)
    points = np.array(points)
    points = np.append(points, np.ones(shape=(len(points), 1)), axis=1)

    rotated_points = rotate_points(points, rotation_origin[1], rotation_origin[0], angle, image.shape[0], image.shape[1])

    rotated_center = rotated_points[0]
    rotated_joints = rotated_points[:, 1:]

    return rotated_image, rotated_center, rotated_joints
# TODO joint visibility

def rotate_points(points, cx, cy, angle, h, w):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    import pdb
    pdb.set_trace()

    calculated = np.dot(M, np.array(points).T).T
    calculated = calculated.reshape(-1, 17)

    return calculated

    # ox, oy = origin
    # angle = np.radians(angle)
    #
    # if len(point) == 2:
    #     px, py = point
    #
    #     qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    #     qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    #     return [qx, qy]
    # else:
    #     px, py, pz = point
    #
    #     qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    #     qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    #     return [qx, qy, pz]

def crop(img, x0, y0, x1, y1):
    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        img, x0, y0, x1, y1 = pad_image(img, x0, y0, x1, y1)
    return img[y0:y1, x0:x1, :]

def pad_image(img, x0, y0, x1, y1):
    img = np.pad(img,
                 ((np.abs(np.minimum(0, y0)), np.maximum(y1 - img.shape[0], 0)),
                  (np.abs(np.minimum(0, x0)), np.maximum(x1 - img.shape[1], 0)),
                  (0,0)), mode="constant")

    x_adjust = np.abs(np.minimum(0, x0))
    y_adjust = np.abs(np.minimum(0, y0))

    x0 += x_adjust
    y0 += y_adjust
    x1 += x_adjust
    y1 += y_adjust
    return img, x0, y0, x1, y1


def flip(image, center, joints):
    flipped_joints = np.copy(joints)

    im_height, im_width, im_channels = image.shape

    flipped_image = cv2.flip(image, flipCode=1) # COMMENT mirrors image x coordinates
    flipped_joints[:, 0] = im_width - joints[:, 0] # COMMENT mirrors joints x coordinates

    for joint_pair in joint_pairs:
        temp = joints[joint_pair[0], :]
        flipped_joints[joint_pair[0], :] = flipped_joints[joint_pair[1], :]
        flipped_joints[joint_pair[1], :] = temp

    flipped_center = center
    flipped_center[0] = im_width - center[0]

    return flipped_image, flipped_center, flipped_joints


def draw_with_joints(image, joints):
    plt.imshow(image)

    for joint in joints:
        plt.scatter(joint[0], joint[1])

    plt.show()


annots_train = []
annots_valid = []

with open(ANNOTATION_JSON_FILENAME) as f:
    json_parsed = json.loads(f.read())

for index, value in enumerate(json_parsed):
    if index != 1: continue
    annots_valid.append(value) if value['isValidation'] == 1.0 else annots_train.append(value)

    process_image(value, True, True, True)

