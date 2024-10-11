import numpy as np
#import green_interpolation
from green_interpolation import green_interpolation
from red_interpolation import red_interpolation
from blue_interpolation import blue_interpolation
import os

def demosaic_function(mosaic_data):
    """
    Main function for the Residual Interpolation demosaicking
    algorithms 'GBTF', 'RI', 'MLRI', 'WMLRI'
    sigma is ignored by GBTF
    """

    # mosaic and mask (just to generate the mask)
    mosaic, mask, pattern = mosaic_data
    
    # imask
    imask = (mask == 0)

    # green interpolation
    green, dif = green_interpolation(mosaic, mask, pattern)

    # parameters for guided upsampling
    h = 5
    v = 5
    eps = 0

    # Red and Blue demosaicking
    red = red_interpolation(green, mosaic, mask, pattern, dif)
    blue = blue_interpolation(green, mosaic, mask, pattern, dif)


    # result image
    rgb_size = mosaic.shape
    rgb_dem = np.zeros((rgb_size[0], rgb_size[1], 3),dtype=np.uint8)
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue

    return rgb_dem




if __name__ == "__main__":
    from skimage.io import imread, imsave
    rgb = imread('Sans_bruit_13.PNG')
    
    rgb = rgb.astype('float32')
    pattern = 'grbg'
    sigma = 1.0
    Algorithm = 'MLRI'

    # generate the mosaic
    mosaic, _ = mosaic_bayer(rgb, pattern)

    # call demosaicing
    rgb_dem = demosaic_RI(mosaic, pattern, sigma, Algorithm)
    imsave('test_%s2.tiff'%Algorithm, rgb_dem.astype('uint8'))
