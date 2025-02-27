import math
import numpy as np


def linefollowing_obj(observation):
    truncated = False

    gray_img = observation.mean(axis=2) ;
    c = observation.shape[1] ;
    black_mask = (gray_img[0,:] < 0.1) ;

    if black_mask.astype(np.int32).sum() < 1: # no black pixels visible
        terminated = True
        return -1.

    x_indices = np.linspace(0.,c,c) ;
    black_indices = x_indices[black_mask] ;
    left_line_index = np.min(black_indices) ;
    
    ret = 1.- math.fabs(left_line_index-c//2) / (c//2) ;

    terminated = ret < 0.
    return ret