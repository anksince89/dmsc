import cv2
import numpy as np

def filter2D(im, ker):
    """
    convolve the 2d  image (im) with the 2d kernel (ker) and return a 2d  image
    pads the image to preserve the shape by replicating boundaries
    """
    return cv2.filter2D(im,  -1, kernel=ker, borderType=cv2.BORDER_REPLICATE)


def boxFilter(im, sz):
    """
    convolve the 2d  image (im) with a box filter of diameter sz (tuple)
    pads the image to preserve the shape by replicating boundaries
    """
    return cv2.boxFilter(im,  -1, sz, normalize=False, borderType=cv2.BORDER_CONSTANT)


def getGaussianKernel(sz,sigma):
    """
    returns a 1d Gaussian kernel with standard deviation sigma and support sz 
    """
    return cv2.getGaussianKernel(sz, sigma)

def get_mosaic_masks(mosaic, pattern):
        """
        generate the mosaic masks assuming a given pattern
        returns:  maskGr, maskGb, maskR, maskB
        """
        size_rawq = mosaic.shape
        maskGr = np.zeros((size_rawq[0], size_rawq[1]))
        maskGb = np.zeros((size_rawq[0], size_rawq[1]))
        maskR  = np.zeros((size_rawq[0], size_rawq[1]))
        maskB  = np.zeros((size_rawq[0], size_rawq[1]))

        if pattern == 'grbg':
            maskGr[0::2, 0::2] = 1
            maskGb[1::2, 1::2] = 1
            maskR [0::2, 1::2] = 1
            maskB [1::2, 0::2] = 1
        elif pattern == 'rggb':
            maskGr[0::2, 1::2] = 1
            maskGb[1::2, 0::2] = 1
            maskB [1::2, 1::2] = 1
            maskR [0::2, 0::2] = 1
        elif pattern == 'gbrg':
            maskGb[0::2, 0::2] = 1
            maskGr[1::2, 1::2] = 1
            maskR [1::2, 0::2] = 1
            maskB [0::2, 1::2] = 1
        elif pattern == 'bggr':
            maskGb[0::2, 1::2] = 1
            maskGr[1::2, 0::2] = 1
            maskB [0::2, 0::2] = 1
            maskR [1::2, 1::2] = 1

        return maskGr, maskGb, maskR, maskB
