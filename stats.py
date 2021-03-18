import utils
import numpy as np

def glas_stats(true_instance_masks, pred_instance_masks):
    count = len(true_instance_masks)

    true_pixel_count = 0
    pred_pixel_count = 0

    for i in range(count):
        true = true_instance_masks[i]
        pred = pred_instance_masks[i]

        true[true > 0] = 1
        pred[pred > 0] = 1

        true_pixel_count += np.sum(true)
        pred_pixel_count += np.sum(pred)

    tp = fp = fn = 0
    dice = haus = 0

    for i in range(count):
        true = true_instance_masks[i]
        pred = pred_instance_masks[i]

        tp_i, fp_i, fn_i = utils.get_tp_fp_fn(pred, true)
        tp += tp_i
        fp += fp_i
        fn += fn_i

        x, y = utils.get_dice_info(pred, true, pred_pixel_count, true_pixel_count)
        dice += 0.5 * (x + y)

        z, w = utils.get_haus_info(pred, true, pred_pixel_count, true_pixel_count)
        haus += 0.5 * (z + w)

    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    f1 = (2 * pr * re) / (pr + re)

    return f1, dice, haus
