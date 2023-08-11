import numpy as np
from .intersection import intersection_over_union
from .box_creators import get_bounding_boxes
from .merger import merge_boxes, merge_two_boxes
from .size_limiter import limit_bounding_boxes


def images_to_cells(binary_images, anchor_boxes, n_col_cells, n_row_cells):
    images_boxes = [get_bounding_boxes(im, padding=1, extra=1) for im in binary_images]
    images_boxes = [merge_boxes(boxes, min_ratio=0.25, min_area=9, n_iter=1) for boxes in images_boxes]
    images_boxes = [limit_bounding_boxes(im, boxes, 24, 24, padding=1, extra=1) for im, boxes in
                    zip(binary_images, images_boxes)]
    images_cells = [bounding_boxes_to_cells(im.shape, boxes, anchor_boxes, n_col_cells, n_row_cells) for im, boxes in
                    zip(binary_images, images_boxes)]
    # images_cells = [normalize_cells(im.shape, cells) for im, cells in zip(binary_images, images_cells)]
    return np.array(images_cells)


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


def normalize_images_cells(image_shape, images_cells):
    images_cells = [normalize_cells(image_shape, cells) for cells in images_cells]
    return np.array(images_cells)


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


def unnormalize_images_cells(image_shape, images_cells):
    images_cells = [unnormalize_cells(image_shape, cells) for cells in images_cells]
    return np.array(images_cells)


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
            x, y, width, height = bbox  # col, row
        # cells[row_cell, col_cell, 0] = 1 # class
        cells[row_cell, col_cell, abox_index*5] = 1
        x = x % cell_width
        y = y % cell_height
        # cells[row_cell, col_cell, coord_low:coord_high] = (x, y, width, height)
        cells[row_cell, col_cell, coord_low:coord_high] = (x/cell_width, y/cell_height, width/cell_width, height/cell_height)

    return cells


