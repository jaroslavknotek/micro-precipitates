---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
---

```python
import pathlib

root = pathlib.Path('../rev-results/20231025215432-crop_size=128-cnn_depth=5-loss=fl-train_loss_denoise_weight=0-val_loss_denoise_weight=0-cnn_filters=8/epoch_70')


import imageio
from tqdm.auto import tqdm

def _l(p):
    return imageio.imread(p)

xs = list(root.rglob('x.png'))
data = { x_path.parent.stem:{'x':_l(x_path),'y':_l(x_path.parent/'y.png'),'foreground':_l(x_path.parent/'foreground.png')}  for x_path in tqdm(xs)}
```

```python
# split in half sometimes ddidn't work. grid is only half-visible or its below half
# training problems
# Line distance errors fixed too (doesn't work when labels are not perfect -> too many 100000 score resutls)
import matplotlib.pyplot as plt

```

```python

```

```python
import logging
logging.basicConfig()
logger = logging.getLogger('palivo')

import numpy as np
import cv2

def angle_between(v1, v2, deg=True):
    """
    Author: Jan Palasek
    Computes angle between two vectors.
    Args:
        v1: First vector. Shape of the vector must be (2,).
        v2: Second vector. Shape of the vector must be (2,).
        deg: If True, return angle in degrees. If False, then the angle will be returned in radians.

    Returns: Angle in degrees (-180, 180] or radians.

    """
    dot = np.dot(v1, v2)
    det = np.cross(v1, v2)

    angle = np.arctan2(det, dot)

    if deg:
        angle = np.rad2deg(angle)

    return angle


def perform_refinement_walk(mask):
    """
    author: Jan Palasek
    Performs walk on the border between black/white components.
    Args:
        mask: Mask to perform walk on.

    Returns: List of coordinates (tuples (height, width)) of the performed path and list of pixels from which the algorithm
    couldn't go further (= graph bridges).

    """
    mask_height, mask_width = mask.shape
    max_val = np.max(mask)

    # find out pixel that are
    first_height = None
    for h in range(mask_height - 1):
        if mask[h, 0] == 0 and mask[h + 1, 0] > 0:
            first_height = h
            break

    assert first_height is not None

    h, w = first_height, 0
    stack = []
    opened = []
    closed = set()
    stack.append((h, w))
    while True:
        h, w = stack[-1]

        # if you are at the end of the image, quit
        if w == mask_width - 1:
            opened.append((h, w))
            break

        # if (h, w) is in closed vertices , ignore it
        # this can happen because we add vertices without checking whether it's already in the stack
        if (h, w) in closed:
            del stack[-1]
            continue

        # if (h, w) is in opened vertices (every its child has been closed already and this vertex is about to be opened again), close it
        if (h, w) in opened:
            del stack[-1]

            assert opened[-1] == (h, w), f"{opened[-1]} is not equal to {(h, w)}"
            del opened[-1]

            closed.add((h, w))
            continue

        if len(opened) > 0:
            prev_h, prev_w = opened[-1]
        else:
            prev_h, prev_w = (h, w)
        prev_direction = (h - prev_h, w - prev_w)

        # otherwise add it to opened vertices
        opened.append((h, w))

        # find next pixel that neighbours white pixel
        # the highest priority is to go to the most right pixel considering what direction we have atm
        # this forces algorithm to explore all possible connected black peaks
        ordered_next_pixels = np.array([
            (h, w - 1), (h + 1, w - 1), (h - 1, w - 1),
            (h + 1, w), (h - 1, w),
            (h, w + 1), (h + 1, w + 1), (h - 1, w + 1)
        ])
        next_directions = [(next_h - h, next_w - w) for next_h, next_w in ordered_next_pixels]
        next_angles = np.array([angle_between(prev_direction, direction) for direction in next_directions])
        sorted_idx = np.argsort(-next_angles)

        ordered_next_pixels = ordered_next_pixels[sorted_idx]

        for next_h, next_w in ordered_next_pixels:
            # invalid coordinates => ignore
            if not (0 <= next_h < mask_height and 0 <= next_w < mask_width):
                continue

            # if the new vertex has already been opened or closed, ignore it
            if (next_h, next_w) in opened or (next_h, next_w) in closed:
                continue

            if mask[next_h, next_w] == 0 and any_neighbour_has_val(mask, next_h, next_w, val=max_val):
                stack.append((next_h, next_w))

    return opened, closed

def clear_invalid_black_pixels( mask):
    
    mask = mask.copy()
    val = np.max(mask)
    _, bridge_pixels = perform_refinement_walk(mask)

    for h, w in bridge_pixels:
        mask[h, w] = val
    return mask

        
def match_problem_points_to_peak_idx(problem_idx, all_peaks_idx):
    """
    Matches problem index into a interval made by nearest peaks.

    Args:
        problem_idx: Indices (width) denoting problems.
        all_peaks_idx: Valid peaks.

    Returns:

    """

    problem_intervals = []
    # for each problem point, find closest interval surrounding it
    for idx in problem_idx:
        # find closest left peak to idx
        left_peak_idx = all_peaks_idx[np.where(all_peaks_idx < idx)]
        left_dist = np.abs(idx - left_peak_idx)
        left_peak = left_peak_idx[np.where(left_dist == np.min(left_dist))][0]

        # find closest right peak to idx
        right_peak_idx = all_peaks_idx[np.where(idx < all_peaks_idx)]

        right_dist = np.abs(idx - right_peak_idx)
        right_peak = right_peak_idx[np.where(right_dist == np.min(right_dist))][0]

        problem_intervals.append((left_peak, right_peak))
    return problem_intervals
    
    
def remove_stains_half(mask: np.ndarray):
# """
#    Author: Jan Palasek
#     This postprocessor splits the images horizontally and postprocesses two result parts separately:
#     upper part of the grid + rods and lower part of the grid + rods.

#     The goal of this postprocessor is to remove so called "stains". Stain is defined as an independent component that
#     is not largest or the second largest in the horizontally split mask.

#     There are two types of stains: white stain on the black background and black stain on the white background.
#     This postprocessor removes both kinds.
# """        
    # find 2 largest components - rods are black, grids are white
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_sizes = stats[:, -1]
    sort_ind = np.argsort(-component_sizes)[:2]

    # set every other component's (not the two largest ones) pixels value as black
    # note that every black pixel is considered as background by cv2 and thus handled as one component
    mask[(labels != sort_ind[0]) & (labels != sort_ind[1])] = 0

    # invert colors - to remove black stains too
    mask = cv2.bitwise_not(mask)

    # find 2 largest components - rods are white, grids are black
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_sizes = stats[:, -1]
    sort_ind = np.argsort(-component_sizes)[:2]

    # set every other than 2 largest component's pixel value as black
    mask[(labels != sort_ind[0]) & (labels != sort_ind[1])] = 0

    # invert pixels back
    return cv2.bitwise_not(mask)

def split_by_mass(mask):
    y_profile = np.sum(mask, axis=1)
    area = np.sum(y_profile)
    cum = y_profile*np.arange(len(y_profile))
    mid = int(np.sum(cum)/area)
    
    mask_top = mask[:mid]
    mask_bottom = mask[mid:]
    
    return mask_top,mask_bottom

def any_neighbour_has_val(mask: np.ndarray, h, w, val):
    """
    Reports whether the pixel on cordinates (h, w) has any neighbour (4-connectivity) with a specified value.
    Args:
        mask: Mask. Every pixel must be 0 or 255.
        h: Height of the coordinate.
        w: Width of the coordinate.
        val: Value to find.

    Returns:

    """
    if h - 1 >= 0 and mask[h - 1, w] == val:
        return True
    if h + 1 < mask.shape[0] and mask[h + 1, w] == val:
        return True
    if w - 1 >= 0 and mask[h, w - 1] == val:
        return True
    if w + 1 < mask.shape[1] and mask[h, w + 1] == val:
        return True

    return False



def _perform_safe(fn,mask):
    try:
        # plt.imshow(mask)
        # plt.title('before')
        res = fn(mask)
        # plt.show()
        # plt.imshow(res)
        # plt.title('after')
        # plt.show()
        return res
    except Exception as e:
        logger.exception(e)
        return mask
        
    
def _threshold_prediction(prediction,thr = .4):
    return np.where(prediction>thr,1,0).astype(np.uint8)
    
def refine_mask_prediction(prediction,assume_full_mask = True):
    # assume_full_mask - If true, algorithm assumes that the prediction is based on a picture of full mask 
    # (has top, bottom and spans from left to right). Meaning that top and bottom line of prediction should be zero
    # and middle should be uninterupted rod of ones
    
    if prediction.dtype == np.uint8 or np.max(prediction) >1:
        prediction = np.float32(prediction)/255
    
    mask = _threshold_prediction(prediction)
    
    mask_top,mask_bottom_teeth_down = split_by_mass(mask)
    # flip mask so it can be processed as the top mask
    mask_bottom = np.flip(mask_bottom_teeth_down,axis=0)
    
    masks = [mask_top,mask_bottom]
    
    #masks is assumed to be left to right, adding artificial prediction
    if assume_full_mask:
        for mask in masks:
            mask[-10:] = 1
            mask[0] = 0

    masks_no_stains = [_perform_safe(remove_stains_half,mask_half) for mask_half in masks]
    masks_no_invalid = [_perform_safe(clear_invalid_black_pixels,mask_half) for mask_half in masks_no_stains]
    
    # remove stains
    
    # walk the teeths
    mask_done_top,mask_done_bottom = masks_no_invalid
    mask_done_bottom_up = np.flip(mask_done_bottom,axis=0)
    return np.vstack([mask_done_top,mask_done_bottom_up])


record = data['29-52-180-g7']
record = data['gKB21_F2_g1']
refined_mask = refine_mask_prediction(record['foreground'])


plt.imshow(refined_mask)
plt.show()
plt.imshow(record['foreground'])
```

```python
import itertools

def line_distance(y_true: np.ndarray, y_pred: np.ndarray, cut_sides = True):
    """
    # Author Jan Palasek
    Computes Line Distance Metric. Performs walk on the border of black and white region for lower and
    upper half for both true and prediction and calculates maximum error on height axis. The input
    prediction MUST be cleaned.

    Args:
        y_true: Shape of (height, width).
        y_pred: Shape of (height, width). Must be cleaned (no stains).

    Returns: Maximum distance from prediction to the true (judged by height).

    """
    assert y_true.shape == y_pred.shape
    
    if cut_sides:
        cut_10p = y_true.shape[1]//10
        y_true =y_true[:,cut_10p:-cut_10p]
        y_pred =y_pred[:,cut_10p:-cut_10p]
    
    # ensure it's the same type
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    upper_half_true , lower_half_true = split_by_mass(y_true)
    upper_half_pred , lower_half_pred = y_pred[:upper_half_true.shape[0]], y_pred[upper_half_true.shape[0]:]
    lower_half_true, lower_half_pred = np.flip(lower_half_true, axis=0), np.flip(lower_half_pred, axis=0)

    try:
        error_lower = np.max(_line_distances_metric_half(lower_half_true, lower_half_pred))
    except Exception as e:
        logger.exception(e)
        error_lower = np.inf
    try:
        error_upper = np.max(_line_distances_metric_half(upper_half_true, upper_half_pred))
    except Exception as e:
        logger.exception(e)
        error_upper = np.inf

    return np.maximum(error_lower, error_upper)


def _line_distances_metric_half(y_true, y_pred):
    # Author Jan Palasek
    """
    Calculates line distances of true from predicted. The distance is a height-distance.
    Args:
        y_true: True mask of shape (image_height, image_width). The values must be 0 or 255.
        y_pred: Prediction of shape (height, width). The values must be 0 or 255.

    Returns: Array of shape (height,), where on every coordinate there is an absolute distance of predicted crossing (curve defining crossing of a predicted mask and a true mask)
    and true crossing.

    """
    true_walk_coordinates,_ = perform_refinement_walk(y_true)
    pred_walk_coordinates,_ = perform_refinement_walk(y_pred)

    return _line_distances_metric_half_from_coords(
        true_walk_coordinates, 
        pred_walk_coordinates
    )

def _line_distances_metric_half_from_coords(true_walk_coordinates, pred_walk_coordinates):
    # Author Jan Palasek
    pred_walk_coordinates = sorted(pred_walk_coordinates, key=lambda x: x[1])
    true_walk_coordinates = sorted(true_walk_coordinates, key=lambda x: x[1])

    errors = []

    f = lambda x: x[1]
    for (k_pred, val_pred), (k_true, val_true) in zip(itertools.groupby(sorted(pred_walk_coordinates, key=f), f),
                                                      itertools.groupby(sorted(true_walk_coordinates, key=f), f)):
        pred_h = np.array(list(x[0] for x in val_pred))
        true_h = np.array(list(x[0] for x in val_true))

        error = []
        for h in pred_h:
            error.append(np.max(np.abs(h - true_h)))
        error = np.max(error)

        errors.append(error)

    return np.array(errors)


```

```python
y_pred = clear
y_true = record['y']


#upper_half_true, upper_half_pred = np.flip(upper_half_true, axis=0), np.flip(upper_half_pred, axis=0)
lower_half_true, lower_half_pred = np.flip(lower_half_true, axis=0), np.flip(lower_half_pred, axis=0)

# for img in [upper_half_pred,lower_half_pred, upper_half_true,lower_half_true]:
#     plt.imshow(img)
#     plt.show()
line_distance(y_true,y_pred)

```

```python
import matplotlib.pyplot as plt
c=4
record = data['29-52-180-g7']
for k,record in data.items():
    foreground = record['foreground']
    clear = refine_mask_prediction(foreground)
    
    y = record['y']
    t,b = split_by_mass(y)
    mask_halfs = [t,np.flip(b,axis=0)]
    t_no_stain,b_no_stains = [_perform_safe(remove_stains_half,mask_half) for mask_half in mask_halfs]
    y=np.vstack([t_no_stain,np.flip(b_no_stains,axis=0)])
    
    ld = line_distance(y,clear)
    fig,axs = plt.subplots(1,4,figsize=(c*4,c))
    axs[0].imshow(y)
    axs[1].imshow(clear)
    axs[2].imshow(foreground)
    axs[3].imshow(record['x'])
    plt.suptitle(f"{k} - {ld:.5f}")
    plt.show()

```

```python

```
