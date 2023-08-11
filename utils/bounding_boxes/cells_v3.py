import numpy as np
from .intersection import intersection_over_union
from .box_creators import get_bounding_boxes
from .merger import merge_boxes
from .size_limiter import limit_bounding_boxes
from .box_utils import to_x_y_width_height


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

