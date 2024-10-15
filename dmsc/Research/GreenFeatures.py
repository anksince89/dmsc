import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import sobel, prewitt
from scipy.signal import convolve2d

class GreenFeatures:
    def __init__(self, image_path, result_folder='result', bayer_pattern='grbg'):
        self.image = cv2.imread(image_path)
        self.result_folder = result_folder
        self.bayer_pattern = bayer_pattern
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.mosaic_img = self.mosaic(self.image)

    def mosaic(self, img):
        # Create mosaic for GRBG Bayer pattern
        mosaic = np.zeros_like(img)
        mosaic[::2, ::2, 1] = img[::2, ::2, 1]  # Green at (0, 0)
        mosaic[::2, 1::2, 0] = img[::2, 1::2, 0]  # Red at (0, 1)
        mosaic[1::2, ::2, 2] = img[1::2, ::2, 2]  # Blue at (1, 0)
        mosaic[1::2, 1::2, 1] = img[1::2, 1::2, 1]  # Green at (1, 1)
        return mosaic

    def save_feature(self, feature_img, feature_name):
        result_path = os.path.join(self.result_folder, f"{feature_name}.bmp")
        cv2.imwrite(result_path, feature_img)
        print(f"Feature {feature_name} saved to {result_path}")

    def horizontal_gradient(self):
        grad_x = np.abs(cv2.Sobel(self.mosaic_img[:, :, 1], cv2.CV_64F, 1, 0, ksize=3))
        self.save_feature(grad_x, "horizontal_gradient")
        return grad_x

    def vertical_gradient(self):
        grad_y = np.abs(cv2.Sobel(self.mosaic_img[:, :, 1], cv2.CV_64F, 0, 1, ksize=3))
        self.save_feature(grad_y, "vertical_gradient")
        return grad_y

    def diagonal_gradient(self):
        grad_d1 = np.abs(sobel(self.mosaic_img[:, :, 1]))  # Diagonal gradient using Sobel filter
        grad_d2 = np.abs(prewitt(self.mosaic_img[:, :, 1]))  # Alternative diagonal gradient using Prewitt
        self.save_feature(grad_d1, "diagonal_gradient_sobel")
        self.save_feature(grad_d2, "diagonal_gradient_prewitt")
        return grad_d1, grad_d2

    def texture_descriptor(self):
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(self.mosaic_img[:, :, 1], n_points, radius, method='uniform')
        self.save_feature(lbp, "texture_descriptor_lbp")
        return lbp

    def green_channel_variance(self):
        green_channel = self.mosaic_img[:, :, 1]
        variance_img = cv2.Laplacian(green_channel, cv2.CV_64F)
        self.save_feature(variance_img, "green_variance")
        return variance_img

    def standard_deviation(self):
        green_channel = self.mosaic_img[:, :, 1]
        std_dev = np.std(green_channel, axis=(0, 1))
        std_img = np.full_like(green_channel, std_dev)
        self.save_feature(std_img, "standard_deviation")
        return std_img

    def clustering_of_similar_intensities(self):
        green_channel = self.mosaic_img[:, :, 1]
        _, thresh = cv2.threshold(green_channel, 127, 255, cv2.THRESH_BINARY)
        self.save_feature(thresh, "intensity_clustering")
        return thresh

    def gradient_magnitude(self):
        grad_x = self.horizontal_gradient()
        grad_y = self.vertical_gradient()
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        self.save_feature(grad_magnitude, "gradient_magnitude")
        return grad_magnitude

    def symmetry_check(self):
        green_channel = self.mosaic_img[:, :, 1]
        flip_hor = np.fliplr(green_channel)
        symmetry = np.abs(green_channel - flip_hor)
        self.save_feature(symmetry, "symmetry_check")
        return symmetry

    def corner_detection(self):
        green_channel = self.mosaic_img[:, :, 1]
        corners = cv2.cornerHarris(green_channel, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        self.save_feature(corners, "corner_detection")
        return corners

    
    def bilinear_interpolation(self):
        r = self.mosaic_img[:,:,0]
        g = self.mosaic_img[:,:,1]
        b = self.mosaic_img[:,:,2]

        # green interpolation
        k_g = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
        convg =convolve2d(g, k_g, 'same')
        g = g + convg

        # red interpolation
        k_r_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
        convr1 =convolve2d(r, k_r_1, 'same')
        convr2 =convolve2d(r+convr1, k_g, 'same')
        r = r + convr1 + convr2

        # blue interpolation
        k_b_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
        convb1 =convolve2d(b, k_b_1, 'same')
        convb2 =convolve2d(b+convb1, k_g, 'same')
        b = b + convb1 + convb2
    
        # Save the interpolated green channel
        self.save_feature(g, "bilinear_interpolation")
        return g

    def process(self):
        self.horizontal_gradient()
        self.vertical_gradient()
        self.diagonal_gradient()
        self.texture_descriptor()
        self.green_channel_variance()
        self.standard_deviation()
        self.clustering_of_similar_intensities()
        self.gradient_magnitude()
        self.symmetry_check()
        self.corner_detection()
        self.bilinear_interpolation()

# Example usage:
image_path = "kodim19.png"
green_interpolator = GreenFeatures(image_path)
green_interpolator.process()
