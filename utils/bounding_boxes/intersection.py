import numpy as np


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

