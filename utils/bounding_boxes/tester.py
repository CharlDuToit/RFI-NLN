import numpy as np
import matplotlib.pyplot as plt
from .box_creators import get_bounding_boxes
from .size_limiter import limit_bounding_boxes
from .merger import merge_boxes
from .box_utils import flatten_images_bounding_boxes, normalize_boxes, unnormalize_boxes
from .drawer import draw_nonzero_cells, draw_grid, draw_bounding_boxes
from .cells import bounding_boxes_to_cells, normalize_cells, unnormalize_cells, cells_to_bounding_boxes


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


def main2():
    from utils.raw_data import load_raw_data
    from utils.data import get_multichannel_patches
    from utils.plotting import save_image_batches_grid
    import pandas as pd
    train_data, train_masks, test_data, test_masks = load_raw_data(
        data_name='HERA_CHARL',
        data_path='/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/',
        # data_path='/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/',
        lofar_subset='all')
    test_masks_patches = get_multichannel_patches(test_masks[0:6, ...], 64, 64, 64, 64)
    # _dir = 'lofar_boxes/test_masks_64_box_p1_e1_first6_merged_limited_merged/'
    _dir = 'hera_boxes/test_masks_64_box_p1_e1_first6_merged_limited/'
    # test_masks_patches = get_multichannel_patches(test_masks, 64, 64, 64, 64)
    # patch_boxes = [get_sweeped_bounding_boxes(mask, min_ratio=0.6, min_area=9) for mask in test_masks_patches]
    patch_boxes = [get_bounding_boxes(mask, padding=1, extra=1) for mask in test_masks_patches]
    patch_boxes = [merge_boxes(boxes, min_ratio=0.6, min_area=9, n_iter=1) for boxes in patch_boxes]
    patch_boxes = [limit_bounding_boxes(mask, boxes, 24, 24, padding=1, extra=1) for mask, boxes in
                   zip(test_masks_patches, patch_boxes)]
    # patch_boxes = [merge_boxes(boxes, min_ratio=0.6, min_area=9, n_iter=1) for boxes in patch_boxes]

    flattened_boxes = flatten_images_bounding_boxes(patch_boxes)
    width_height_dict = dict(width=flattened_boxes[:, 2], height=flattened_boxes[:, 3])
    df = pd.DataFrame.from_dict(width_height_dict)
    df.to_csv(_dir + 'lofar_width_heights_test_64.csv', index=False)
    patch_mask_and_boxes_images = np.array(
        [draw_bounding_boxes(mask, box) for mask, box in zip(test_masks_patches, patch_boxes)])
    # patch_mask_and_boxes_images = np.array(
    #    [draw_grid(im, 8, 8) for im in patch_mask_and_boxes_images])
    save_image_batches_grid(_dir, patch_mask_and_boxes_images)

    # save_image_batches_grid('./lofar_boxes', test_masks)
    # save_image_batches_grid('./lofar_boxes', test_masks_patches)


def main():
    import pandas as pd

    im = dummy_image()
    plt.imsave('dummy.png', im)
    # corners, annotations = get_corners(im)
    bboxes = get_bounding_boxes(im, padding=1, extra=1)
    bboxes = limit_bounding_boxes(im, bboxes, 16, 16, padding=1, extra=1)
    # bboxes = get_sweeped_bounding_boxes(im)
    color_im = draw_bounding_boxes(im, bboxes)
    # color_im = draw_grid(color_im, 10, 10)
    # x_y_width_height_dict = dict(x=[b[0] for b in bboxes],
    #                              y=[b[1] for b in bboxes],
    #                              width=[b[2] for b in bboxes],
    #                              height=[b[3] for b in bboxes])
    # df = pd.DataFrame.from_dict(x_y_width_height_dict)
    # df.to_csv('./dummy_x_y_width_height.csv', index=False)
    plt.imsave('dummy_boxes.png', color_im)

    aboxes = dummy_anchor_boxes()
    cells = bounding_boxes_to_cells(bboxes, aboxes, 20, 10)
    cells = normalize_cells(im.shape, cells)
    cell_im = draw_nonzero_cells(cells)
    plt.imsave('dummy_cells.png', cell_im)
    cells = unnormalize_cells(im.shape, cells)
    bb = cells_to_bounding_boxes(cells)
    bb = normalize_boxes(im.shape, bb)
    bb = unnormalize_boxes(im.shape, bb)
    color_im = draw_bounding_boxes(im, bb)
    plt.imsave('dummy_boxes_test.png', color_im)

    # print(bb)
    # for i in range(10):
    #     for j in range(10):
    #         print(i, j, ': ', cells[i, j, :])


if __name__ == '__main__':
    main2()

