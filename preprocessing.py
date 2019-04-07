import scipy.misc
from scipy.ndimage import rotate as rot
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import cv2


def draw_processed_image(image, obj_center, obj_joints, scale, angle, draw_bbox=True):
    plt.imshow(image)

    plt.scatter(obj_center[0], obj_center[1], c='r')

    if draw_bbox:
        x0 = int(obj_center[0] - 200 * scale / 2)
        y0 = int(obj_center[1] - 200 * scale / 2)
        x1 = int(obj_center[0] + 200 * scale / 2)
        y1 = int(obj_center[1] + 200 * scale / 2)

        plt.plot([x0, x1], [y0, y0], c='b')
        plt.plot([x0, x1], [y1, y1], c='b')
        plt.plot([x0, x0], [y0, y1], c='b')
        plt.plot([x1, x1], [y0, y1], c='b')

    plt.title('Scale = {}\nAngle = {}'.format(scale, angle))

    for joint in obj_joints:
        plt.scatter(joint[0], joint[1], c='g')

    plt.show()

def rotate(image, obj_center, obj_joints, angle):
    rotated_image = rot(image, angle)
    rotation_origin = (image.shape[0] // 2, image.shape[1] // 2)

    points = np.copy(obj_center)
    points = points.reshape(-1, 2)
    points = np.append(points, obj_joints, axis=0)
    points = np.append(points, np.ones(shape=(len(points), 1)), axis=1)

    rotated_points = rotate_points(points, rotation_origin[1], rotation_origin[0], angle, image.shape[0], image.shape[1])

    rotated_center = rotated_points[0]
    rotated_joints = rotated_points[1:, :]

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

    calculated = np.dot(points, M.T)
    return calculated

def crop(img, obj_center, obj_joints, scale):
    x0 = int(obj_center[0] - 200 * scale / 2)
    y0 = int(obj_center[1] - 200 * scale / 2)
    x1 = int(obj_center[0] + 200 * scale / 2)
    y1 = int(obj_center[1] + 200 * scale / 2)

    if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
        img, x0, y0, x1, y1 = pad_image(img, x0, y0, x1, y1)

    obj_center[0] -= x0
    obj_center[1] -= y0
    obj_joints[:, 0] -= x0
    obj_joints[:, 1] -= y0

    return img[y0:y1, x0:x1, :], obj_center, obj_joints

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


def flip(image, center, joints, joint_pairs):
    flipped_joints = np.copy(joints)

    im_height, im_width, im_channels = image.shape

    flipped_image = cv2.flip(image, flipCode=1) # COMMENT mirrors image x coordinates
    flipped_joints[:, 0] = im_width - joints[:, 0] # COMMENT mirrors joints x coordinates

    for joint_pair in joint_pairs:
        temp = np.copy(flipped_joints[joint_pair[0], :])
        flipped_joints[joint_pair[0], :] = flipped_joints[joint_pair[1], :]
        flipped_joints[joint_pair[1], :] = temp

    flipped_center = np.copy(center)
    flipped_center[0] = im_width - center[0]

    return flipped_image, flipped_center, flipped_joints