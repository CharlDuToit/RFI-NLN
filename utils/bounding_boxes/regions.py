import numpy as np
import matplotlib.pyplot as plt


def dummy_image():
    im = np.zeros((100, 200), dtype=int)
    im[0:5, 0:5] = 1  # 10 10
    im[10:20, 10:20] = 1  # 10 10
    im[30:35, 10:20] = 1  # 5 10
    im[30:35, 25:30] = 1  # 5 5
    im[40:45, 30:50] = 1  # 5 20
    im[30:50, 55:60] = 1  # 20 5  aaaa
    im[58:60, 60:62] = 1  # 2 2
    im[62:64, 64:74] = 1  # 2 10
    im[70:80, 60:62] = 1  # 10 2
    im[85:90, 70:100] = 1  # 5 30

    # Cross
    im[5:35, 83:88] = 1  # 30 5
    im[18:22, 70:100] = 1  # 5 30

    # Flat regions and single pixel
    im[99:100, 0:5] = 1  # 1 5
    im[90:95, 0:1] = 1  # 5 1
    im[88:89, 4:5] = 1  # 5 1

    return im


def dummy_anchor_boxes():
    anchor_boxes = []
    anchor_boxes.append((10, 10))  # 0
    anchor_boxes.append((15, 15))  # 1
    anchor_boxes.append((35, 35))  # 2
    anchor_boxes.append((20, 10))  # 3
    anchor_boxes.append((30, 10))  # 4
    anchor_boxes.append((15, 10))  # 5
    anchor_boxes.append((10, 30))  # 6
    anchor_boxes.append((10, 15))  # 7
    anchor_boxes = np.array(anchor_boxes, dtype=float)
    return anchor_boxes


def intersection_over_union(width_1, height_1, width_2, height_2):
    width = np.minimum(width_1, width_2)
    height = np.minimum(height_1, height_2)
    inter = width * height
    union = (width_1 * height_1) + (width_2 * height_2) - inter
    return inter / union


def intersection(box_1, box_2):
    x_overlap = max(0,
                    min(box_1[0] + box_1[2] / 2,
                        box_2[0] + box_2[2] / 2) - max(box_1[0] - box_1[2] / 2,
                                                       box_2[0] - box_2[2] / 2)
                    )
    y_overlap = max(0,
                    min(box_1[1] + box_1[3] / 2,
                        box_2[1] + box_2[3] / 2) - max(box_1[1] - box_1[3] / 2,
                                                       box_2[1] - box_2[3] / 2)
                    )
    inter = x_overlap * y_overlap
    return inter


def intersection_over_min_area(box_1, box_2):
    inter = intersection(box_1, box_2)
    area_1 = box_1[2] * box_1[3]
    area_2 = box_2[2] * box_2[3]
    return inter / min(area_1, area_2)


def get_bounding_boxes(binary_image, padding=1, extra=0):
    # If extra > padding then a box can have pixels from neighouring boxes

    # Convert the binary image to a numpy array
    if binary_image.shape[-1] in (1, 2):
        binary_array = np.array(binary_image[..., 0])
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


def zero_box(binary_image, box):
    left, top, right, bottom = to_left_top_right_bottom(box)
    if binary_image.shape[-1] in (1, 2):
        binary_array = np.array(binary_image[..., 0])
    else:
        binary_array = np.array(binary_image)
    binary_array[top:bottom + 1, left:right + 1] = 0
    return binary_array


def one_box(binary_image, box):
    left, top, right, bottom = to_left_top_right_bottom(box)
    if binary_image.shape[-1] in (1, 2):
        binary_array = np.array(binary_image[..., 0])
    else:
        binary_array = np.array(binary_image)
    binary_array[top:bottom + 1, left:right + 1] = 1
    return binary_array


def get_region(binary_image, box):
    left, top, right, bottom = to_left_top_right_bottom(box)
    if binary_image.shape[-1] in (1, 2):
        binary_array = np.array(binary_image[..., 0])
    else:
        binary_array = np.array(binary_image)
    return binary_array[top:bottom + 1, left:right + 1]


def get_sweeped_bounding_boxes(binary_image, min_ratio=0.6, min_area=9):
    # Convert the binary image to a numpy array
    if binary_image.shape[-1] in (1, 2):
        binary_array = np.array(binary_image[..., 0])
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


def add_extra(image_shape, boundary_boxes, extra):
    if boundary_boxes.shape[0] == 0:
        return boundary_boxes
    for i in range(boundary_boxes.shape[0]):
        left, top, right, bottom = to_left_top_right_bottom(boundary_boxes[i])
        top = max(0, top - extra)
        left = max(0, left - extra)
        bottom = min(image_shape[0] - 1, bottom + extra)
        right = min(image_shape[1] - 1, right + extra)
        boundary_boxes[i] = to_x_y_width_height((left, top, right, bottom))
    return boundary_boxes


#
# def remove_extra(boundary_boxes, extra):
#     boundary_boxes[:, 2] -= 2 * extra
#     boundary_boxes[:, 3] -= 2 * extra
#     return boundary_boxes


def normalize_boxes(image_shape, bounding_boxes):
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
    bounding_boxes[:, 0] *= image_shape[1]
    bounding_boxes[:, 1] *= image_shape[0]
    bounding_boxes[:, 2] *= image_shape[1]
    bounding_boxes[:, 3] *= image_shape[0]
    return bounding_boxes  # returned boundary_boxes has same reference as input boundary_boxes


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


def merge_two_boxes(box_1, box_2):
    # Must be un-normalized
    left_1, top_1, right_1, bottom_1 = to_left_top_right_bottom(box_1)
    left_2, top_2, right_2, bottom_2 = to_left_top_right_bottom(box_2)
    left = min(left_1, left_2)
    top = min(top_1, top_2)
    right = max(right_1, right_2)
    bottom = max(bottom_1, bottom_2)
    return to_x_y_width_height((left, top, right, bottom))

    # merged_box = (
    #     min(box_1[0] - box_1[2] / 2, box_2[0] - box_2[2] / 2),
    #     min(box_1[1] - box_1[3] / 2, box_2[1] - box_2[3] / 2),
    #     max(box_1[0] + box_1[2] / 2, box_2[0] + box_2[2] / 2) - min(box_1[0] - box_1[2] / 2,
    #                                                                           box_2[0] - box_2[2] / 2),
    #     max(box_1[1] + box_1[3] / 2, box_2[1] + box_2[3] / 2) - min(box_1[1] - box_1[3] / 2,
    #                                                                           box_2[1] - box_2[3] / 2)
    # )



def bounding_boxes_to_cells(image_shape, bounding_boxes, anchor_boxes, n_col_cells, n_row_cells):
    # Assume x,y of boundary boxes
    # boxes and anchors must be un-normalized
    # 5 values = is object in anchor box + 4 coordinates
    # returns normalized cells
    n_anchors = anchor_boxes.shape[0]
    cells = np.zeros((n_row_cells, n_col_cells, n_anchors * 5), dtype=np.float32)
    # cells = np.zeros((n_row_cells, n_col_cells, n_anchors * 5 + 1), dtype=np.float32)

    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for i, bbox in enumerate(bounding_boxes):
        abox_index = np.argmax([intersection_over_union(bbox[2], bbox[3], abox[0], abox[1]) for abox in anchor_boxes])
        x, y, width, height = bbox  # col, row
        col_cell = int(x // cell_width)
        row_cell = int(y // cell_height)
        coord_low = 5 * abox_index + 1
        coord_high = coord_low + 4
        if cells[row_cell, col_cell, abox_index*5]:
            print(f'Bounding box {i} at cell {row_cell}, {col_cell} already has bb assigned to ab of type {abox_index}. Merging boxes')
            x_old, y_old, width_old, height_old = cells[row_cell, col_cell, abox_index*5+1:abox_index*5+5]
            x_old, y_old = (col_cell + x_old) * cell_width, (row_cell + y_old) * cell_height
            width_old, height_old = width_old * cell_width, height_old * cell_height
            bbox = merge_two_boxes(bbox, (x_old, y_old, width_old, height_old))
            x, y, width, height = bbox
        # cells[row_cell, col_cell, 0] = 1 # class
        cells[row_cell, col_cell, abox_index*5] = 1
        x = x % cell_width
        y = y % cell_height
        # cells[row_cell, col_cell, coord_low:coord_high] = (x, y, width, height)
        cells[row_cell, col_cell, coord_low:coord_high] = (x/cell_width, y/cell_height, width/cell_width, height/cell_height)

    return cells


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

    if binary_image.shape[-1] in (1, 2):
        binary_array = np.array(binary_image[..., 0])
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


def normalize_cells(image_shape, cells):
    n_anchors = int(cells.shape[2] / 5)
    x_inds = []
    y_inds = []
    w_inds = []
    h_inds = []
    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for i in range(n_anchors):
        x_inds.append(i * 4 + n_anchors)
        y_inds.append(i * 4 + 1 + n_anchors)
        w_inds.append(i * 4 + 2 + n_anchors)
        h_inds.append(i * 4 + 3 + n_anchors)
    cells[:, :, x_inds] /= cell_width
    cells[:, :, y_inds] /= cell_height
    cells[:, :, w_inds] /= image_shape[1]
    cells[:, :, h_inds] /= image_shape[0]
    return cells


def unnormalize_cells(image_shape, cells):
    n_anchors = int(cells.shape[2] / 5)
    x_inds = []
    y_inds = []
    w_inds = []
    h_inds = []
    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for i in range(n_anchors):
        x_inds.append(i * 4 + n_anchors)
        y_inds.append(i * 4 + 1 + n_anchors)
        w_inds.append(i * 4 + 2 + n_anchors)
        h_inds.append(i * 4 + 3 + n_anchors)
    cells[:, :, x_inds] *= cell_width
    cells[:, :, y_inds] *= cell_height
    cells[:, :, w_inds] *= image_shape[1]
    cells[:, :, h_inds] *= image_shape[0]
    return cells


def cells_to_bounding_boxes(image_shape, cells, threshold=0.5):
    # Assume to be unnormalized
    n_anchors = int(cells.shape[2] / 5)
    rows, cols, aboxes = np.nonzero(cells[:, :, [5*i for i in range(n_anchors)]] > threshold)
    boxes = []

    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for r, c, ab_index in zip(rows, cols, aboxes):
        coord_low = 4 * ab_index + 1
        coord_high = coord_low + 4
        box = cells[r, c, coord_low:coord_high]
        box[0] = c * cell_width + box[0] * cell_width
        box[1] = r * cell_height + box[1] * cell_height
        box[2] = box[2] * cell_width
        box[3] = box[3] * cell_height
        boxes.append(box)
    return np.array(boxes)


def images_cells_to_images_bounding_boxes(image_shape, images_cells, threshold=0.5):
    # Assume to be unnormalized
    images_boundary_boxes = [None] * images_cells.shape[0]
    for i, cells in enumerate(images_cells):
        images_boundary_boxes[i] = cells_to_bounding_boxes(image_shape, cells, threshold)
    return images_boundary_boxes


def draw_nonzero_cells(cells, threshold=0.5):
    n_anchors = int(cells.shape[2] / 5)
    rows, cols, aboxes = np.nonzero(cells[:, :, 0:n_anchors] > threshold)
    im = np.zeros(cells.shape[0:2])
    im[rows, cols] = 1
    return im


def draw_grid(image, n_rows, n_cols):
    if image.shape[-1] in (1, 2):
        color_image = np.stack((image[..., 0] / 3,) * 3, axis=-1)
    elif image.shape[-1] in (3,):  # already a color image
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
    if binary_image.shape[-1] in (1, 2):
        color_image = np.stack((binary_image[..., 0] / 3,) * 3, axis=-1)
    elif binary_image.shape[-1] in (3,):  # already a color image
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
    if binary_image.shape[-1] in (1, 2):
        color_image = np.stack((binary_image[..., 0] / 3,) * 3, axis=-1)
    elif binary_image.shape[-1] in (3,):  # already a color image
        color_image = np.array(binary_image[...])
    else:
        color_image = np.stack((binary_image / 3,) * 3, axis=-1)

    # Draw a red boundary box around each region of interest
    for box in bounding_boxes:
        # left, top, right, bottom = box
        left, top, right, bottom = to_left_top_right_bottom(box)
        top = max(0, top)
        left = max(0, left)
        right = min(color_image.shape[0], right)
        top = min(color_image.shape[1], top)

        color_image[top:bottom + 1, left] = [1, 0, 0]
        color_image[top:bottom + 1, right] = [1, 0, 0]
        color_image[top, left:right + 1] = [1, 0, 0]
        color_image[bottom, left:right + 1] = [1, 0, 0]

    for box in true_bounding_boxes:
        # left, top, right, bottom = box
        left, top, right, bottom = to_left_top_right_bottom(box)
        top = max(0, top)
        left = max(0, left)
        right = min(color_image.shape[0], right)
        top = min(color_image.shape[1], top)

        color_image[top:bottom + 1, left] = [0, 0, 1]
        color_image[top:bottom + 1, right] = [0, 0, 1]
        color_image[top, left:right + 1] = [0, 0, 1]
        color_image[bottom, left:right + 1] = [0, 0, 1]

    return color_image.astype(float)


def flatten_images_bounding_boxes(images_bounding_boxes):
    result = []
    for boxes in images_bounding_boxes:
        for box in boxes:
            result.append(box)
    return np.array(result)


def images_to_cells(binary_images, anchor_boxes, n_col_cells, n_row_cells):
    images_boxes = [get_bounding_boxes(im, padding=1, extra=1) for im in binary_images]
    images_boxes = [merge_boxes(boxes, min_ratio=0.25, min_area=9, n_iter=1) for boxes in images_boxes]
    images_boxes = [limit_bounding_boxes(im, boxes, 24, 24, padding=1, extra=1) for im, boxes in
                    zip(binary_images, images_boxes)]
    images_cells = [bounding_boxes_to_cells(im.shape, boxes, anchor_boxes, n_col_cells, n_row_cells) for im, boxes in
                    zip(binary_images, images_boxes)]
    return np.array(images_cells)


def create_anchor_boxes(sizes=(5, 12, 24)):
    anchor_boxes = []
    n_sizes = len(sizes)
    for i in range(n_sizes):
        for j in range(n_sizes):
            anchor_boxes.append((sizes[i], sizes[j]))
    anchor_boxes = np.array(anchor_boxes, dtype=float)
    return anchor_boxes



def images_to_cells_v2(binary_images, anchor_boxes, n_col_cells, n_row_cells):
    images_boxes = [get_bounding_boxes(im, padding=1, extra=1) for im in binary_images]
    images_boxes = [merge_boxes(boxes, min_ratio=0.25, min_area=9, n_iter=1) for boxes in images_boxes]
    images_boxes = [limit_bounding_boxes(im, boxes, 8, 8, padding=1, extra=1) for im, boxes in
                    zip(binary_images, images_boxes)]
    images_cells = [bounding_boxes_to_cells_v2(im.shape, boxes, anchor_boxes, n_col_cells, n_row_cells) for im, boxes in
                    zip(binary_images, images_boxes)]
    # images_cells = [normalize_cells(im.shape, cells) for im, cells in zip(binary_images, images_cells)]
    return np.array(images_cells)

#
# def normalize_cells(image_shape, cells):
#     n_anchors = int(cells.shape[2] / 5)
#     x_inds = []
#     y_inds = []
#     w_inds = []
#     h_inds = []
#     cell_width = int(image_shape[1] / cells.shape[1])
#     cell_height = int(image_shape[0] / cells.shape[0])
#     for i in range(n_anchors):
#         x_inds.append(i * 4 + n_anchors)
#         y_inds.append(i * 4 + 1 + n_anchors)
#         w_inds.append(i * 4 + 2 + n_anchors)
#         h_inds.append(i * 4 + 3 + n_anchors)
#     cells[:, :, x_inds] /= cell_width
#     cells[:, :, y_inds] /= cell_height
#     cells[:, :, w_inds] /= image_shape[1]
#     cells[:, :, h_inds] /= image_shape[0]
#     return cells
#
#
# def normalize_images_cells(image_shape, images_cells):
#     images_cells = [normalize_cells(image_shape, cells) for cells in images_cells]
#     return np.array(images_cells)
#
#
# def unnormalize_cells(image_shape, cells):
#     n_anchors = int(cells.shape[2] / 5)
#     x_inds = []
#     y_inds = []
#     w_inds = []
#     h_inds = []
#     cell_width = int(image_shape[1] / cells.shape[1])
#     cell_height = int(image_shape[0] / cells.shape[0])
#     for i in range(n_anchors):
#         x_inds.append(i * 4 + n_anchors)
#         y_inds.append(i * 4 + 1 + n_anchors)
#         w_inds.append(i * 4 + 2 + n_anchors)
#         h_inds.append(i * 4 + 3 + n_anchors)
#     cells[:, :, x_inds] *= cell_width
#     cells[:, :, y_inds] *= cell_height
#     cells[:, :, w_inds] *= image_shape[1]
#     cells[:, :, h_inds] *= image_shape[0]
#     return cells
#
#
# def unnormalize_images_cells(image_shape, images_cells):
#     images_cells = [unnormalize_cells(image_shape, cells) for cells in images_cells]
#     return np.array(images_cells)


def cells_to_bounding_boxes_v2(image_shape, cells, threshold=0.5):
    # Assumes cells are normalized
    # returns unnormalized boxes
    n_anchors = int(cells.shape[2] / 5)
    rows, cols, aboxes = np.nonzero(cells[:, :, 0:n_anchors] > threshold)
    boxes = []

    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for r, c, ab_index in zip(rows, cols, aboxes):
        coord_low = 4 * ab_index + n_anchors
        coord_high = coord_low + 4
        box = cells[r, c, coord_low:coord_high]
        box[0] = c * cell_width + box[0] * cell_width
        box[1] = r * cell_height + box[1] * cell_height
        box[2] = box[2] * cell_width
        box[3] = box[3] * cell_height
        boxes.append(box)
    return np.array(boxes)


def images_cells_to_images_bounding_boxes_v2(image_shape, images_cells, threshold=0.5):
    # Assume cells are normalized
    # returns list of unnormalized bounding boxes for each image
    images_boundary_boxes = [None] * images_cells.shape[0]
    for i, cells in enumerate(images_cells):
        images_boundary_boxes[i] = cells_to_bounding_boxes_v2(image_shape, cells, threshold)
    return images_boundary_boxes


def bounding_boxes_to_cells_v2(image_shape, bounding_boxes, anchor_boxes, n_col_cells, n_row_cells):
    # assigns and normalizes relative to cell
    # Assume x,y, width, height of boundary boxes as integers
    # boxes and anchors must be un-normalized
    # 5 values = is object in anchor box + 4 coordinates
    # First n_anchors values in each cell indicates if that anchor box is present
    n_anchors = anchor_boxes.shape[0]
    cells = np.zeros((n_row_cells, n_col_cells, n_anchors * 5), dtype=np.float32)

    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for i, bbox in enumerate(bounding_boxes):
        abox_index = np.argmax([intersection_over_union(bbox[2], bbox[3], abox[0], abox[1]) for abox in anchor_boxes])
        x, y, width, height = bbox  # col, row
        col_cell = int(x // cell_width)
        row_cell = int(y // cell_height)
        if cells[row_cell, col_cell, abox_index]:
            print(f'Bounding box {i} at cell {row_cell}, {col_cell} already has bb assigned to ab of type {abox_index}')
        cells[row_cell, col_cell, abox_index] = 1
        coord_low = 4 * abox_index + n_anchors
        coord_high = coord_low + 4
        x = x % cell_width
        y = y % cell_height
        cells[row_cell, col_cell, coord_low:coord_high] = (x/cell_width, y/cell_height, width/cell_width, height/cell_height)

    return cells


def cells_to_bounding_boxes_v3(image_shape, cells, threshold=0.25):
    # Assumes cells are normalized
    # returns unnormalized boxes in col, row, width, height format
    # n_anchors = int(cells.shape[2] / 5)
    n_anchors = int(cells.shape[2] / 6)
    rows, cols, aboxes = np.nonzero(cells[:, :, 0:n_anchors] > threshold)
    boxes = []

    cell_width = int(image_shape[1] / cells.shape[1])
    cell_height = int(image_shape[0] / cells.shape[0])
    for r, c, ab_index in zip(rows, cols, aboxes):
        coord_low = 4 * ab_index + n_anchors + 1
        coord_high = coord_low + 4
        box = cells[r, c, coord_low:coord_high]
        box[0] = c * cell_width + box[0] * cell_width
        box[1] = r * cell_height + box[1] * cell_height
        box[2] = box[2] * cell_width
        box[3] = box[3] * cell_height
        boxes.append(box)
    return np.array(boxes)


def images_cells_to_images_bounding_boxes_v3(image_shape, images_cells, threshold=0.5):
    # Assume cells are normalized
    # returns list of unnormalized bounding boxes for each image
    images_boundary_boxes = [None] * images_cells.shape[0]
    for i, cells in enumerate(images_cells):
        images_boundary_boxes[i] = cells_to_bounding_boxes_v3(image_shape, cells, threshold)
    return images_boundary_boxes


def images_masks_to_cells(images_masks, n_row_cells, n_col_cells):
    cells = np.zeros((images_masks.shape[0], n_row_cells, n_col_cells, 6), dtype=np.float32)
    # cells = np.zeros((images_masks.shape[0], n_row_cells, n_col_cells, 5), dtype=np.float32)
    for i, mask in enumerate(images_masks):
        cells[i, ...] = mask_to_cells(mask, n_row_cells, n_col_cells)
    return cells


def mask_to_cells(mask, n_row_cells, n_col_cells):
    # Divides mask into cells and calculates boundary box if there is one

    cell_width = int(mask.shape[1] / n_col_cells)
    cell_height = int(mask.shape[0] / n_row_cells)
    # cells = np.zeros((n_row_cells, n_col_cells, 5), dtype=np.float32)
    cells = np.zeros((n_row_cells, n_col_cells, 6), dtype=np.float32)

    for col_cell in range(n_col_cells):
        for row_cell in range(n_row_cells):
            sub_mask = mask[row_cell * cell_height:(row_cell + 1) * cell_height,
                            col_cell * cell_width:(col_cell + 1) * cell_width, ...].reshape((cell_height, cell_width))
            rows, cols = np.nonzero(sub_mask)
            if len(rows) == 0:
                continue
            left, right, top, bottom = np.min(cols), np.max(cols), np.min(rows), np.max(rows)
            x, y, width, height = to_x_y_width_height((left, top, right, bottom))
            # cells[row_cell, col_cell, 0:6] = [1.0,
            #                                   x/cell_width,
            #                                   y/cell_height,
            #                                   width/cell_width,
            #                                   height/cell_height]
            cells[row_cell, col_cell, 0:7] = [1.0,
                                              1.0,
                                              x/cell_width,
                                              y/cell_height,
                                              width/cell_width,
                                              height/cell_height]
    return cells




def main2():
    from utils.raw_data import load_raw_data
    from utils.data import get_multichannel_patches
    from utils.plotting import save_image_batches_grid
    import pandas as pd
    import os
    train_data, train_masks, test_data, test_masks = load_raw_data(
        data_name='LOFAR',
        # data_path='/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/',
        data_path='/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/',
        lofar_subset='all')
    test_masks_patches = get_multichannel_patches(test_masks[0:6, ...], 64, 64, 64, 64)

    _dir = 'lofar_boxes/test_masks_v2/'

    if not os.path.exists(_dir):
        os.makedirs(_dir)

    patches_cells = images_to_cells(test_masks_patches, np.array([[4, 4]]), 8, 8)
    patches_boxes = images_cells_to_images_bounding_boxes(test_masks_patches.shape[1:], patches_cells, 0.5)

    color_images = draw_images_bounding_boxes(test_masks_patches, patches_boxes)
    save_image_batches_grid(_dir, color_images)


def main():
    import pandas as pd

    im = dummy_image()
    plt.imsave('dummy.png', im)
    # corners, annotations = get_corners(im)
    bboxes = get_bounding_boxes(im, padding=1, extra=1)
    bboxes = limit_bounding_boxes(im, bboxes, 16, 16, padding=1, extra=1)
    # bboxes = get_sweeped_bounding_boxes(im)
    color_im = draw_bounding_boxes(im, bboxes)
    color_im = draw_grid(color_im, 10, 20)
    # x_y_width_height_dict = dict(x=[b[0] for b in bboxes],
    #                              y=[b[1] for b in bboxes],
    #                              width=[b[2] for b in bboxes],
    #                              height=[b[3] for b in bboxes])
    # df = pd.DataFrame.from_dict(x_y_width_height_dict)
    # df.to_csv('./dummy_x_y_width_height.csv', index=False)
    plt.imsave('dummy_boxes.png', color_im)

    aboxes = dummy_anchor_boxes()
    # cells = bounding_boxes_to_cells(im.shape, bboxes, aboxes, 20, 10)
    cells = bounding_boxes_to_cells_v2(im.shape, bboxes, np.array([[4, 4]]), 20, 10)
    # cells = normalize_cells(im.shape, cells)
    cell_im = draw_nonzero_cells(cells)
    plt.imsave('dummy_cells.png', cell_im)
    # cells = unnormalize_cells(im.shape, cells)
    bb = cells_to_bounding_boxes_v2(im.shape, cells, 0.5)
    #bb = normalize_boxes(im.shape, bb)
    #bb = unnormalize_boxes(im.shape, bb)
    color_im = draw_bounding_boxes(im, bb)
    plt.imsave('dummy_boxes_test.png', color_im)

    # print(bb)
    # for i in range(10):
    #     for j in range(10):
    #         print(i, j, ': ', cells[i, j, :])


def main3():
    from utils.raw_data import load_raw_data
    from utils.data import get_multichannel_patches
    from utils.plotting import save_image_batches_grid
    import pandas as pd
    import os
    train_data, train_masks, test_data, test_masks = load_raw_data(
        data_name='LOFAR',
        # data_path='/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/',
        data_path='/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/',
        lofar_subset='all')
    test_masks_patches = get_multichannel_patches(test_masks[0:6, ...], 64, 64, 64, 64)
    _dir = 'lofar_boxes/test_masks_v3/'
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    patches_cells = images_masks_to_cells(test_masks_patches, 8, 8)
    patches_bb = images_cells_to_images_bounding_boxes_v3(test_masks_patches.shape[1:], patches_cells, 0.5)

    patches_with_bb = draw_images_bounding_boxes(test_masks_patches, patches_bb)
    save_image_batches_grid(_dir, patches_with_bb)


if __name__ == '__main__':
    main3()
