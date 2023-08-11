import numpy as np


def normalize_boxes(image_shape, bounding_boxes):
    if bounding_boxes.shape[0] == 0:
        return bounding_boxes
    bounding_boxes[:, 0] /= image_shape[1]
    bounding_boxes[:, 1] /= image_shape[0]
    bounding_boxes[:, 2] /= image_shape[1]
    bounding_boxes[:, 3] /= image_shape[0]
    return bounding_boxes  # returned boundary_boxes has same reference as input boundary_boxes


def normalize_anchors(image_shape, anchor_boxes):
    anchor_boxes[:, 0] /= image_shape[1]
    anchor_boxes[:, 1] /= image_shape[0]
    return anchor_boxes  # returned boundary_boxes has same reference as input boundary_boxes


def unnormalize_boxes(image_shape, bounding_boxes):
    if bounding_boxes.shape[0] == 0:
        return bounding_boxes
    bounding_boxes[:, 0] *= image_shape[1]
    bounding_boxes[:, 1] *= image_shape[0]
    bounding_boxes[:, 2] *= image_shape[1]
    bounding_boxes[:, 3] *= image_shape[0]
    return bounding_boxes  # returned boundary_boxes has same reference as input boundary_boxes


# def unnormalize_images_boxes(image_shape, bounding_boxes):
#     if bounding_boxes.shape[0] == 0:
#         return bounding_boxes
#     bounding_boxes[:, 0] *= image_shape[1]
#     bounding_boxes[:, 1] *= image_shape[0]
#     bounding_boxes[:, 2] *= image_shape[1]
#     bounding_boxes[:, 3] *= image_shape[0]
#     return bounding_boxes  # returned boundary_boxes has same reference as input boundary_boxes


def to_left_top_right_bottom(x_y_width_height):
    # x_y_width_height must be unnormalized
    x, y, width, height = x_y_width_height
    top = np.ceil(y - height / 2).astype(int)
    bottom = np.floor(y + height / 2).astype(int)
    right = np.floor(x + width / 2).astype(int)
    left = np.ceil(x - width / 2).astype(int)
    return left, top, right, bottom


def to_x_y_width_height(left_top_right_bottom):
    left, top, right, bottom = left_top_right_bottom
    x = (left + right) / 2  # x is based on columns
    y = (top + bottom) / 2  # y is based on rows
    width = right - left + 1
    height = bottom - top + 1
    return x, y, width, height


def zero_box(binary_image, box):
    left, top, right, bottom = to_left_top_right_bottom(box)
    if len(binary_image.shape) > 2:
        binary_array = np.array(binary_image[:, :, 0])
    else:
        binary_array = np.array(binary_image)
    binary_array[top:bottom + 1, left:right + 1] = 0
    return binary_array


def one_box(binary_image, box):
    left, top, right, bottom = to_left_top_right_bottom(box)
    if len(binary_image.shape) > 2:
        binary_array = np.array(binary_image[:, :, 0])
    else:
        binary_array = np.array(binary_image)
    binary_array[top:bottom + 1, left:right + 1] = 1
    return binary_array


def get_region(binary_image, box):
    left, top, right, bottom = to_left_top_right_bottom(box)
    if len(binary_image.shape) > 2:
        binary_array = np.array(binary_image[:, :, 0])
    else:
        binary_array = np.array(binary_image)
    return binary_array[top:bottom + 1, left:right + 1]


def add_extra(image_shape, boundary_boxes, extra):
    if boundary_boxes.shape[0] == 0:
        return boundary_boxes
    for i in range(boundary_boxes.shape[0]):
        left, top, right, bottom = to_left_top_right_bottom(boundary_boxes[i])
        top = max(0, top - extra)
        left = max(0, left - extra)
        bottom = min(image_shape[0] - 1, bottom + extra)
        right = min(image_shape[1] - 1, right + extra)
        # if bottom > 63:
            # print('aa')
        boundary_boxes[i] = to_x_y_width_height((left, top, right, bottom))
    return boundary_boxes


def flatten_images_bounding_boxes(images_bounding_boxes):
    result = []
    for boxes in images_bounding_boxes:
        for box in boxes:
            result.append(box)
    return np.array(result)


def create_anchor_boxes(sizes=(5, 12, 24)):
    anchor_boxes = []
    n_sizes = len(sizes)
    for i in range(n_sizes):
        for j in range(n_sizes):
            anchor_boxes.append((sizes[i], sizes[j]))
    anchor_boxes = np.array(anchor_boxes, dtype=float)
    return anchor_boxes
