import numpy as np
from .box_utils import to_left_top_right_bottom, to_x_y_width_height


def draw_nonzero_cells(cells, threshold=0.5):
    n_anchors = int(cells.shape[2] / 5)
    rows, cols, aboxes = np.nonzero(cells[:, :, 0:n_anchors] > threshold)
    im = np.zeros(cells.shape[0:2])
    im[rows, cols] = 1
    return im


def draw_grid(image, n_rows, n_cols):
    if len(image.shape) > 2 and image.shape[2] == 1:
        color_image = np.stack((image[:, :, 0] / 3,) * 3, axis=-1)
    elif len(image.shape) > 2 and image.shape[2] == 3:
        color_image = np.array(image[...])
    else:
        color_image = np.stack((image / 3,) * 3, axis=-1)

    for r in range(0, color_image.shape[0], int(color_image.shape[0] // n_rows)):
        if r < color_image.shape[0]:
            color_image[r, :] = [0, 1, 0]
    for c in range(0, color_image.shape[1], int(color_image.shape[1] // n_cols)):
        if c < color_image.shape[1]:
            color_image[:, c] = [0, 1, 0]
    return color_image


def draw_images_bounding_boxes(binary_images, images_bounding_boxes):
    color_images = [None] * binary_images.shape[0]
    i = 0
    for im, bboxes in zip(binary_images, images_bounding_boxes):
        color_images[i] = draw_bounding_boxes(im, bboxes)
        i += 1
    return np.array(color_images)


def draw_bounding_boxes(binary_image, bounding_boxes):
    # Convert the binary image to a color image
    if len(binary_image.shape) > 2 and binary_image.shape[2] == 1:
        color_image = np.stack((binary_image[:, :, 0] / 3,) * 3, axis=-1)
    elif len(binary_image.shape) > 2 and binary_image.shape[2] == 3:
        color_image = np.array(binary_image[...])
    else:
        color_image = np.stack((binary_image / 3,) * 3, axis=-1)

    # Draw a red boundary box around each region of interest
    for box in bounding_boxes:
        # left, top, right, bottom = box
        left, top, right, bottom = to_left_top_right_bottom(box)
        top = min(max(0, top), color_image.shape[1]-1)
        left = min(max(0, left), color_image.shape[0]-1)
        right = max(min(color_image.shape[0]-1, right), 0)
        bottom = max(min(color_image.shape[1]-1, bottom), 0)

        color_image[top:bottom + 1, left] = [1, 0, 0]
        color_image[top:bottom + 1, right] = [1, 0, 0]
        color_image[top, left:right + 1] = [1, 0, 0]
        color_image[bottom, left:right + 1] = [1, 0, 0]

    return color_image.astype(float)


def draw_bounding_boxes_true_pred(binary_image, bounding_boxes, true_bounding_boxes):
    # Convert the binary image to a color image
    if len(binary_image.shape) > 2 and binary_image.shape[2] == 1:
        color_image = np.stack((binary_image[:, :, 0] / 3,) * 3, axis=-1)
    elif len(binary_image.shape) > 2 and binary_image.shape[2] == 3:
        color_image = np.array(binary_image[...])
    else:
        color_image = np.stack((binary_image / 3,) * 3, axis=-1)

    # Draw a red boundary box around each region of interest
    for box in bounding_boxes:
        # left, top, right, bottom = box
        left, top, right, bottom = to_left_top_right_bottom(box)
        top = max(0, top)
        left = max(0, left)
        right = min(color_image.shape[0]-1, right)
        bottom = min(color_image.shape[1]-1, bottom)

        color_image[top:bottom + 1, left] = [1, 0, 0]
        color_image[top:bottom + 1, right] = [1, 0, 0]
        color_image[top, left:right + 1] = [1, 0, 0]
        color_image[bottom, left:right + 1] = [1, 0, 0]

    for box in true_bounding_boxes:
        # left, top, right, bottom = box
        left, top, right, bottom = to_left_top_right_bottom(box)
        top = max(0, top)
        left = max(0, left)
        right = min(color_image.shape[0]-1, right)
        bottom = min(color_image.shape[1]-1, top)

        color_image[top:bottom + 1, left] = [0, 0, 1]
        color_image[top:bottom + 1, right] = [0, 0, 1]
        color_image[top, left:right + 1] = [0, 0, 1]
        color_image[bottom, left:right + 1] = [0, 0, 1]

    return color_image.astype(float)

