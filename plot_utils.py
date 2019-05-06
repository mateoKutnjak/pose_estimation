import matplotlib.pyplot as plt

def plot_labelmaps(original_image, original_joints, labelmaps, labelmap_joints):
    fig, ax = plt.subplots(nrows=4, ncols=5)

    ax[0, 0].imshow(original_image)
    ax[0, 0].scatter(original_joints[:, 0], original_joints[:, 1])

    for i in range(0, 4):
        for j in range(1, 5):
            ax[i, j].imshow(labelmaps[:, :, i*4 + j-1])
            ax[i, j].scatter(labelmap_joints[:, 0], labelmap_joints[:, 1], s=1, c='r')

    plt.show()

def plot_processed_image(image, obj_joints, obj_center=None, scale=1.0, angle=0.0, draw_bbox=True):
    plt.imshow(image)

    if obj_center is not None:
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

def plot_predicted_joints(image, obj_joints, output_shape, save_filename=None):
    scale = image.shape[0] / output_shape[0]

    for joint in obj_joints:
        plt.scatter(scale * joint[0], scale * joint[1], c='g')

    plt.imshow(image)

    if save_filename is None:
        plt.show()
    else:
        plt.savefig(save_filename, bbox_inches='tight')