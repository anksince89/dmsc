import numpy as np
from utils import filter2D




def red_interpolation(green, mosaic, mask, pattern, dif):
    """ 
    red interpolation implementing Residual Interpolation demosaicking
    algorithms ('GBTF', 'RI', 'MLRI', 'WMLRI')  
    Arguments: 
        green: image containing the interpolated green channel
        mosaic: 3 channel image containing the R G B mosaic
        mask: 3 channel image indicating where the mosaic is set
        pattern: Bayer pattern 'grbg', 'rggb', 'gbrg', 'bggr'
        h,v: support of the guided filter
        eps: guided filter regularization (use 0) 
        dif: green residual image (from RIXgreen_interpolation)
        Algorithm: one of 'GBTF', 'RI', 'MLRI', 'WMLRI'
    Returns: 
        red: the interpolated red channel 
    """
    # This functions implements Algorithm 4
    Prb = np.array([[0, 0, -1, 0, -1, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0], 
                    [-1, 0, 10, 0, 10, 0, -1],
                    [0, 0, 0, 0, 0, 0, 0], 
                    [-1, 0, 10, 0, 10, 0, -1], 
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, -1, 0, 0]]) / 32
    Aknl = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4

    # this line corresponds to line 4 of Algorithm 4
    red = mosaic[:, :, 0] + mask[:, :, 2] * (green - filter2D(dif, Prb))
    # this line computes:  G - [\hat G - \hat R] \otimes K_A 
    tempimg = mosaic[:, :, 1] - mask[:, :, 1] * filter2D(green, Aknl)\
              + mask[:, :, 1] * filter2D(red, Aknl)
    red = red + tempimg

    # R interpolation
    red = np.clip(red, 0, 255).astype(np.uint8)

    return red



