"""Evaluation code for instance segmentation as used in the GlaS challenge.

https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/

Provide the paths to the results and ground truth directories at `pred_dir` 
and `true_dir` respectively. Each directory should contain a single file per
image, where each object should be given a unique id. 

To run, use python compute_stats.py

Author: Simon Graham

"""

import glob
import os

from utils import *

def compute_stats(pred_dir, true_dir):
    pred_ext = ".npy"
    true_ext = ".npy"

    file_list = glob.glob('%s/*%s' % (pred_dir, pred_ext))
    file_list.sort() # ensure same order

    num_ims = len(file_list)

    tp = 0
    fp = 0
    fn = 0  
    odice_tmp1 = 0
    odice_tmp2 = 0
    ohaus_tmp1 = 0
    ohaus_tmp2 = 0

    haus = 0

    """
    Toggle whether or not to calculate the below stats
    Warning: object hausdorff takes a while!
    """
    obj_dice = True
    obj_f1 = True

    y_true = []
    y_prob = []

    if obj_f1:
        total_pix_pred, total_pix_true = total_pix(pred_dir, true_dir, pred_ext, true_ext)

    metrics = {}

    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        #basename = basename.split('_')[0] + '_' + basename.split('_')[1]

        print('Obtaining stats for ',basename)

        true = np.load(true_dir + basename + '.npy')
        true = true.astype('int32')

        pred = np.load(pred_dir + basename + '.npy')
        pred = pred.astype('int32')

        z, w = get_haus_info(pred, true, total_pix_pred, total_pix_true)
        haus += 0.5 * (z + w)

        if obj_f1:
            # Get info for calculating object f1
            tp_, fp_, fn_ = get_tp_fp_fn(pred, true)
            tp += tp_
            fp += fp_
            fn += fn_

        if obj_dice:
            # Get info for calculating object dice
            odice_tmp1_, odice_tmp2_ = get_dice_info(pred, true, total_pix_pred, total_pix_true)
            odice_tmp1 += odice_tmp1_
            odice_tmp2 += odice_tmp2_


    #------------------------------------------------------------------------------------------


    if obj_f1:
        # Calculate object F1
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        obj_f1 = (2 * pr * re)  / (pr + re)
        metrics['Obj F1'] = obj_f1

    if obj_dice:
        # Calculate object dice
        obj_dice = 0.5 * (odice_tmp1 + odice_tmp2)
        metrics['Obj Dice'] = obj_dice

    metrics['Obj Haus'] = haus


    return metrics



