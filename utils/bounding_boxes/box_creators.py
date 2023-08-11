
import numpy as np
from .box_utils import to_x_y_width_height
from .merger import merge_to_existing_boxes


def get_bounding_boxes(binary_image, padding=1, extra=0):
    # If extra > padding then a box can have pixels from neighouring boxes

    # Convert the binary image to a numpy array
    if len(binary_image.shape) > 2:
        binary_array = np.array(binary_image[:, :, 0])
    else:
        binary_array = np.array(binary_image)
    # binary_array_orig = np.array(binary_array)

    # Find the rows and columns where the binary image is non-zero
    non_zero_rows, non_zero_cols = np.nonzero(binary_array)

    # Determine the bounding boxes for each contiguous region of non-zero pixels
    bounding_boxes = []
    for true_ind in range(len(non_zero_rows)):
        row = non_zero_rows[true_ind]
        col = non_zero_cols[true_ind]

        # Check if the pixel has already been processed
        if binary_array[row][col] == 0:
            continue

        # Initialize the bounding box coordinates
        top = row
        left = col
        bottom = row
        right = col

        # Update the bounding box coordinates for all pixels in the region
        stack = [(row, col)]
        while len(stack) > 0:
            r, c = stack.pop()
            if binary_array[r][c] == 0:
                continue
            top = min(top, r)
            left = min(left, c)
            bottom = max(bottom, r)
            right = max(right, c)

            binary_array[r][c] = 0

            for i in range(max(0, r - padding), min(binary_array.shape[0], r + padding + 1)):
                for j in range(max(0, c - padding), min(binary_array.shape[1], c + padding + 1)):
                    if binary_array[i][j] == 1:
                        stack.append((i, j))

        # Adjust the bounding box coordinates
        top = max(0, top - extra)
        left = max(0, left - extra)
        bottom = min(binary_array.shape[0] - 1, bottom + extra)
        right = min(binary_array.shape[1] - 1, right + extra)

        x, y, width, height = to_x_y_width_height((left, top, right, bottom))

        # Add the bounding box to the list
        # bounding_boxes.append((left, top, right, bottom))
        bounding_boxes.append((x, y, width, height))

    return np.array(bounding_boxes)


def get_sweeped_bounding_boxes(binary_image, min_ratio=0.6, min_area=9):
    # Convert the binary image to a numpy array
    if len(binary_image.shape) > 2:
        binary_array = np.array(binary_image[:, :, 0])
    else:
        binary_array = np.array(binary_image)
    # binary_array_orig = np.array(binary_array)

    # Find corners
    corners, annotations = get_corners(binary_array)

    # Determine the bounding boxes for each contiguous region of non-zero pixels
    bounding_boxes = []
    for corn, ann in zip(corners, annotations):
        hor_box = hor_sweep_box(binary_array, corn, ann)
        bounding_boxes = merge_to_existing_boxes(hor_box, bounding_boxes, min_ratio, min_area)
        ver_box = ver_sweep_box(binary_array, corn, ann)
        bounding_boxes = merge_to_existing_boxes(ver_box, bounding_boxes, min_ratio, min_area)

    return np.array(bounding_boxes)


def hor_sweep_box(binary_array, corner, corner_annotation):
    row, col = corner[0], corner[1]
    # Horizontal sweep
    left = col
    right = col
    current_col = col
    current = 1
    if 'L' in corner_annotation:
        dcol = 1
    elif 'R' in corner_annotation:
        dcol = -1
    else:
        raise ValueError('Corner annotation missing R and L')

    while current == 1:
        left = min(left, current_col)
        right = max(right, current_col)
        current_col += dcol
        if current_col == -1 or current_col == binary_array.shape[1]:
            current = 0
        else:
            current = binary_array[row, current_col]
    # Vertical sweep from left and right
    top = row
    bottom = row
    current_row = row
    current = 1
    if 'T' in corner_annotation:
        drow = 1
    elif 'B' in corner_annotation:
        drow = -1
    else:
        raise ValueError('Corner annotation missing T and B')

    while current == 1:
        top = min(top, current_row)
        bottom = max(bottom, current_row)
        current_row += drow
        if current_row == -1 or current_row == binary_array.shape[0]:
            current = 0
        else:
            current = min(binary_array[current_row, left], binary_array[current_row, right])

    # Create horizontal sweep box
    box = to_x_y_width_height((left, top, right, bottom))
    return box


def ver_sweep_box(binary_array, corner, corner_annotation):
    row, col = corner[0], corner[1]
    # Vertical sweep
    top, bottom, current_row, current = row, row, row, 1
    if 'T' in corner_annotation:
        drow = 1
    elif 'B' in corner_annotation:
        drow = -1
    else:
        raise ValueError('Corner annotation missing T and B')

    while current == 1:
        top = min(top, current_row)
        bottom = max(bottom, current_row)
        current_row += drow
        if current_row == -1 or current_row == binary_array.shape[0]:
            current = 0
        else:
            current = binary_array[current_row, col]

    # Horizontal sweep from top and bottom
    left, right, current_col, current = col, col, col, 1
    if 'L' in corner_annotation:
        dcol = 1
    elif 'R' in corner_annotation:
        dcol = -1
    else:
        raise ValueError('Corner annotation missing R and L')

    while current == 1:
        left = min(left, current_col)
        right = max(right, current_col)
        current_col += dcol
        if current_col == -1 or current_col == binary_array.shape[1]:
            current = 0
        else:
            current = min(binary_array[top, current_col], binary_array[bottom, current_col])

    # Create vertical sweep box
    box = to_x_y_width_height((left, top, right, bottom))
    return box


def get_corners(binary_array):
    # if binary_image.shape[-1] in (1, 2):
    #     binary_array = np.array(binary_image[..., 0])
    # else:
    #     binary_array = np.array(binary_image)
    # binary_array_orig = np.array(binary_array)

    # Find the rows and columns where the binary image is non-zero
    non_zero_rows, non_zero_cols = np.nonzero(binary_array)

    # Find corners
    corners = []
    annotations = []
    for row, col in zip(non_zero_rows, non_zero_cols):
        ann = ''
        if row == 0 or binary_array[row - 1, col] == 0:
            ann += 'T'
        if row == binary_array.shape[0] - 1 or binary_array[row + 1, col] == 0:
            ann += 'B'
        if col == 0 or binary_array[row, col - 1] == 0:
            ann += 'L'
        if col == binary_array.shape[1] - 1 or binary_array[row, col + 1] == 0:
            ann += 'R'
        if ('T' in ann or 'B' in ann) and ('R' in ann or 'L' in ann):
            corners.append((row, col))
            annotations.append(ann)

    return np.array(corners), annotations

