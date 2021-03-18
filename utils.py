import numpy as np 
import scipy
import os
import glob
from scipy.ndimage.measurements import label
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment


def label_im(im):
    struct = [[1,1,1],
              [1,1,1],
              [1,1,1]]
    return label(im, structure=struct)[0]


# def total_pix(pred_dir, true_dir, pred_ext, true_ext, idx=None):
def total_pix(pred_dir, true_dir, pred_ext, true_ext, idx=None):

    num_pred = 0
    num_true = 0

    file_list = glob.glob(pred_dir + '*' + pred_ext)
    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        if true_ext[-3:] == 'npy':
            true = np.load(true_dir + basename + true_ext)
        else:
            true = cv2.imread(true_dir + basename + true_ext, 0)
        true = true.astype('int32')
        true[true>0] = 1
        num_true += np.sum(true)

        if pred_ext[-3:] == 'npy':
            pred = np.load(pred_dir + basename + pred_ext)
        else:
            pred = cv2.imread(pred_dir + basename + pred_ext, 0)
        pred = pred.astype('int32')
        pred[pred>0] = 1
        num_pred += np.sum(pred)

    return num_pred, num_true


def hausdorff_distance(pred, true):

    coord_pred = np.where(pred)
    coord_pred = np.vstack(coord_pred).transpose()

    coord_true = np.where(true)
    coord_true = np.vstack(coord_true).transpose()

    haus1 = directed_hausdorff(coord_pred, coord_true)[0]

    haus2 = directed_hausdorff(coord_true, coord_pred)[0]
    haus_dist = max(haus1, haus2)

    return haus_dist


def find_nearest(mask, candidates):

    min_dst = 10000
    for idx in range(1, len(candidates)):
        candidate = candidates[idx]
        candidate = candidate.astype('int32')
        dst = hausdorff_distance(mask, candidate)
        if dst < min_dst:
            min_dst = dst

    return min_dst


def get_tp_fp_fn(pred, true):
    
    tp = 0
    fp = 0
    fn = 0

    true = np.copy(true) # get copy of GT
    pred = np.copy(pred) # get copy of prediction
    true = true.astype('int32')
    pred = pred.astype('int32')
    pred = label_im(pred)
    true = label_im(true)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    num_gt = true_id.copy()
    num_gt.remove(0)
    num_gt = len(num_gt)

    # Check to see whether there exists any predictions / GT instances
    if len(pred_id) == 1 and len(true_id) == 1:
        tp = 0
        fp = 0
        fn = 0
    elif len(pred_id) == 1:
        tp = 0
        fp = 0
        fn = len(true_id) - 1
    elif len(true_id) == 1:
        tp = 0
        fp = len(pred_id) -1
        fn = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for pred_idx in range(1, len(pred_id)):
        p_mask = pred_masks[pred_idx]
        true_pred_overlap = true[p_mask > 0]
        true_pred_overlap_id = np.unique(true_pred_overlap)
        true_pred_overlap_id = list(true_pred_overlap_id)
        try: # remove background
            true_pred_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(true_pred_overlap_id) > 0:
            t_mask_combine = np.zeros(p_mask.shape)
            for true_idx in true_pred_overlap_id:
                t_mask_combine += true_masks[true_idx]
            inter = (t_mask_combine * p_mask).sum()
            size_gt = t_mask_combine.sum()
            if inter / size_gt > 0.5:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = num_gt - tp
    
    return tp, fp, fn


def get_dice_info(pred, true, total_pix_pred, total_pix_true):
    d_temp1 = 0
    d_temp2 = 0

    true = np.copy(true) # get copy of GT
    pred = np.copy(pred) # get copy of prediction
    true = true.astype('int32')
    pred = pred.astype('int32')
    pred = label_im(pred)
    true = label_im(true)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    num_gt = true_id.copy()
    num_gt.remove(0)
    num_gt = len(num_gt)

    # Check to see whether there exists any predictions / GT instances
    if len(pred_id) == 1 and len(true_id) == 1:
        tp = 0
        fp = 0
        fn = 0
    elif len(pred_id) == 1:
        tp = 0
        fp = 0
        fn = len(true_id) - 1
    elif len(true_id) == 1:
        tp = 0
        fp = len(pred_id) -1
        fn = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # First term in RHS of object dice equation
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        t_mask_area = np.sum(t_mask)
        gamma = t_mask_area / total_pix_true
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try: # remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(pred_true_overlap_id) > 0:
            inter_max = 0
            for pred_idx in pred_true_overlap_id:
                inter = (pred_masks[pred_idx] * t_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    pred_idx_max = pred_idx
            p_mask_max = pred_masks[pred_idx_max]
            total = p_mask_max.sum() + t_mask.sum()
            d_temp1 += (gamma * 2 * inter_max) / total  # calculate dice (gamme gives more weight to larger glands in the GT)
    
    # Second term in RHS of object dice equation
    for pred_idx in range(1, len(pred_id)):
        p_mask = pred_masks[pred_idx]
        p_mask_area = np.sum(p_mask)
        sigma = p_mask_area / total_pix_pred
        true_pred_overlap = true[p_mask > 0]
        true_pred_overlap_id = np.unique(true_pred_overlap)
        true_pred_overlap_id = list(true_pred_overlap_id)
        try: # remove background
            true_pred_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(true_pred_overlap_id) > 0:
            inter_max = 0
            for true_idx in true_pred_overlap_id:
                inter = (true_masks[true_idx] * p_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    true_idx_max = true_idx
            t_mask_max = true_masks[true_idx_max]
            total = t_mask_max.sum() + p_mask.sum()
            d_temp2 += (sigma * 2 * inter_max) / total  # calculate dice (gamme gives more weight to larger glands in the GT)
            
    return d_temp1, d_temp2


def get_haus_info(pred, true, total_pix_pred, total_pix_true):
    h_temp1 = 0
    h_temp2 = 0

    true = np.copy(true) # get copy of GT
    pred = np.copy(pred) # get copy of prediction
    true = true.astype('int32')
    pred = pred.astype('int32')
    pred = label_im(pred)
    true = label_im(true)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    num_gt = true_id.copy()
    num_gt.remove(0)
    num_gt = len(num_gt)

    # Check to see whether there exists any predictions / GT instances
    if len(pred_id) == 1 and len(true_id) == 1:
        tp = 0
        fp = 0
        fn = 0
    elif len(pred_id) == 1:
        tp = 0
        fp = 0
        fn = len(true_id) - 1
    elif len(true_id) == 1:
        tp = 0
        fp = len(pred_id) -1
        fn = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    
    # First term in RHS of object hausdorff equation
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        t_mask_area = np.sum(t_mask)
        gamma = t_mask_area / total_pix_true
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try: # remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(pred_true_overlap_id) > 0:
            inter_max = 0
            for pred_idx in pred_true_overlap_id:
                inter = (pred_masks[pred_idx] * t_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    pred_idx_max = pred_idx
            p_mask_max = pred_masks[pred_idx_max]
            haus_dist = hausdorff_distance(t_mask, p_mask_max)
            h_temp1 += gamma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)
        else:
            haus_dist = find_nearest(t_mask, pred_masks)  # if no matching predition, use closest prediction to corresponding GT
            h_temp1 += gamma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)
    
    # Second term in RHS of object hausdorff equation
    for pred_idx in range(1, len(pred_id)):
        p_mask = pred_masks[pred_idx]
        p_mask_area = np.sum(p_mask)
        sigma = p_mask_area / total_pix_pred
        true_pred_overlap = true[p_mask > 0]
        true_pred_overlap_id = np.unique(true_pred_overlap)
        true_pred_overlap_id = list(true_pred_overlap_id)
        try: # remove background
            true_pred_overlap_id.remove(0)
        except ValueError:
            pass  # just means there's no background
        if len(true_pred_overlap_id) > 0:
            inter_max = 0
            for true_idx in true_pred_overlap_id:
                inter = (true_masks[true_idx] * p_mask).sum()
                if inter > inter_max:
                    inter_max = inter
                    true_idx_max = true_idx
            t_mask_max = true_masks[true_idx_max]
            haus_dist = hausdorff_distance(t_mask_max, p_mask)
            h_temp2 += sigma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)
        else:
            haus_dist = find_nearest(p_mask, true_masks)  # if no matching GT, use closest GT to corresponding prediction
            h_temp2 += sigma * haus_dist  # calculate hausforff distance (gamme gives more weight to larger glands in the GT)

    return h_temp1, h_temp2


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger instances has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred

