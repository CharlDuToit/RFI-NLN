import numpy as np
from .intersection import intersection_over_min_area
from .box_utils import to_x_y_width_height, to_left_top_right_bottom


def merge_to_existing_boxes(box, existing_boxes, min_ratio=0.6, min_area=9):
    # box is not in existing_boxes
    # box will either be appended to existing_boxes or will be merged into an existing box
    merge = False
    for i, existing_box in enumerate(existing_boxes):
        inter_ratio = intersection_over_min_area(box, existing_box)
        if inter_ratio > 0.0:
            if min(box[2] * box[3], existing_box[2] * existing_box[3]) < min_area:
                merge = True
            elif inter_ratio > min_ratio:
                merge = True
            if merge:
                existing_boxes[i] = merge_two_boxes(box, existing_box)
                break
    if not merge:
        # existing_boxes.append(box)
        existing_boxes = np.row_stack([existing_boxes, [box]])
    return np.array(existing_boxes)


def merge_boxes(bounding_boxes, min_ratio=0.6, min_area=9, n_iter=1):
    n_boxes = bounding_boxes.shape[0]
    if n_boxes == 0:
        return bounding_boxes
    for iter_count in range(n_iter):
        n_boxes = bounding_boxes.shape[0]
        for i in range(n_boxes - 1, -1, -1):
            box = bounding_boxes[i]
            inds = [a for a in range(len(bounding_boxes))]
            inds.remove(i)
            bounding_boxes = merge_to_existing_boxes(box, bounding_boxes[inds], min_ratio, min_area)
    return bounding_boxes


def merge_two_boxes(box_1, box_2):
    # Must be un-normalized
    left_1, top_1, right_1, bottom_1 = to_left_top_right_bottom(box_1)
    left_2, top_2, right_2, bottom_2 = to_left_top_right_bottom(box_2)
    left = min(left_1, left_2)
    top = min(top_1, top_2)
    right = max(right_1, right_2)
    bottom = max(bottom_1, bottom_2)
    return to_x_y_width_height((left, top, right, bottom))




