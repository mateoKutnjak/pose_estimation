import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def heatmap_batch_accuracy(prediction_batch, metadata_batch, threshold):
    positives, negatives = 0, 0

    for i in range(prediction_batch.shape[0]):
        heatmap = prediction_batch[i, :, :, :]

        _positives, _negatives = heatmap_accuracy(heatmap, metadata_batch[i], norm=6.4, threshold=threshold)

        positives += _positives
        negatives += _negatives

    return positives, negatives

def heatmap_accuracy(heatmap, metadata, norm, threshold):
    predicted_joints = find_heatmap_joints(heatmap, threshold)
    labeled_joints = metadata['obj_joints']

    correct_pred, wrong_pred = 0, 0

    for i in range(labeled_joints.shape[0]):
        within_threshold = joints_distance(
            predicted_joints[i, :],
            labeled_joints[i, :],
            norm,
            threshold
        )

        if within_threshold is None: continue

        if within_threshold:
            correct_pred += 1
        else:
            wrong_pred += 1

    return correct_pred, wrong_pred

def find_heatmap_joints(heatmap, threshold):
    joints_positions = []

    for i in range(heatmap.shape[-1]):
        heatmap_layer = heatmap[:, :, i]
        heatmap_layer = gaussian_filter(heatmap_layer, sigma=0.5)

        peaks = non_max_suppression(heatmap_layer)

        y, x = np.where(peaks == peaks.max())

        if len(x) > 0 and len(y) > 0:
            joints_positions.append( (int(x[0]), int(y[0]), peaks[x[0], y[0]]) )
        else:
            joints_positions.append( (0, 0, 0) )
    return np.array(joints_positions)


def non_max_suppression(heatmap_layer, window_size=3, threshold=1e-6):
    under_threshold_indices = heatmap_layer < threshold
    heatmap_layer[under_threshold_indices] = 0

    return heatmap_layer * (heatmap_layer == maximum_filter(heatmap_layer, size=(window_size, window_size)))

def joints_distance(predicted_joint, labeled_joint, norm, threshold):
    if labeled_joint[0] > 1 and labeled_joint[1] > 1:
        distance = np.linalg.norm(predicted_joint[0:2] - labeled_joint[0:2]) / norm
        return distance < threshold
    return None