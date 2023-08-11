import numpy as np
from .box_utils import to_left_top_right_bottom, add_extra
from .box_creators import get_bounding_boxes


def divide_into_intervals(N, max_width):
    N = int(N)
    if N <= max_width:
        return np.array([(0, N - 1)])

    num_intervals = int(N // max_width)
    remaining = N % max_width
    start = 0

    if remaining > 0:
        num_intervals += 1

    width = int(N / num_intervals)
    remaining = N % width
    intervals = []
    for i in range(num_intervals):
        if i < remaining:
            end = start + width
            # end = start + max_width - 1
        else:
            end = start + width - 1
            # end = start + max_width - 2
        intervals.append((start, end))
        start = end + 1
    return np.array(intervals)


def limit_one_bounding_box(binary_array, bounding_box, max_width, max_height, padding=1, extra=0):
    # Splits bounding box into smaller boxes
    if bounding_box[2] < max_width and bounding_box[3] < max_height:
        return np.array([bounding_box])

    # if binary_image.shape[-1] in (1, 2):
    #     binary_array = np.array(binary_image[..., 0])
    # else:
    #     binary_array = np.array(binary_image)

    left, top, right, bottom = to_left_top_right_bottom(bounding_box)

    row_intervals = divide_into_intervals(bounding_box[3], max_height) + top
    col_intervals = divide_into_intervals(bounding_box[2], max_width) + left

    smaller_bounding_boxes = None
    for row_interval in row_intervals:
        to, bo = row_interval[0], row_interval[1]
        for col_interval in col_intervals:
            le, ri = col_interval[0], col_interval[1]
            im = binary_array[to:bo + 1, le:ri + 1]
            bboxes = get_bounding_boxes(im, padding=padding, extra=0)  # Use 0 extra now
            if bboxes.shape[0] == 0:
                continue  # There are no boxes
            bboxes[:, 0] += le
            bboxes[:, 1] += to
            if smaller_bounding_boxes is None:
                smaller_bounding_boxes = bboxes
            else:
                smaller_bounding_boxes = np.row_stack([smaller_bounding_boxes, bboxes])
    return add_extra(binary_array.shape, smaller_bounding_boxes, extra)


def limit_bounding_boxes(binary_image, bounding_boxes, max_width, max_height, padding=1, extra=0):
    if bounding_boxes.shape[0] == 0:
        return bounding_boxes

    if len(binary_image.shape) > 2:
        binary_array = np.array(binary_image[:, :, 0])
    else:
        binary_array = np.array(binary_image)

    smaller_bounding_boxes = None
    for bounding_box in bounding_boxes:
        bboxes = limit_one_bounding_box(binary_array, bounding_box, max_width, max_height, padding, extra)
        if smaller_bounding_boxes is None:
            smaller_bounding_boxes = bboxes
        else:
            smaller_bounding_boxes = np.row_stack([smaller_bounding_boxes, bboxes])
    return smaller_bounding_boxes

