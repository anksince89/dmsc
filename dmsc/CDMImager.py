import os
import cv2
import numpy as np
import csv
import sys
import importlib.util
from skimage.metrics import structural_similarity as ssim

class CDMImager:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.input_folder = os.path.join("data", dataset_name, "GT")
        self.result_folder = os.path.join("data", dataset_name, f"result_{dataset_name}")
        self.demosaicker_folder = "Demosaicker"
        self.bayer_type = 'grbg'
        
        # Create result folder if it doesn't exist
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

    def mosaic_bayer(self,rgb, pattern):
        """
        generate a mosaic from a rgb image
        pattern can be: 'grbg', 'rggb', 'gbrg', 'bggr'
        """
        num = np.zeros(len(pattern))
        pattern_list = list(pattern)
        print(pattern_list)
        p = pattern_list.index('r')
        num[p] = 0
        p = [idx for idx, i in enumerate(pattern_list) if i == 'g']
        num[p] = 1
        p = pattern_list.index('b')
        num[p] = 2

        size_rgb = rgb.shape
        mask = np.zeros((size_rgb[0], size_rgb[1], 3))

        # Generate mask
        mask[0::2, 0::2, int(num[0])] = 1
        mask[0::2, 1::2, int(num[1])] = 1
        mask[1::2, 0::2, int(num[2])] = 1
        mask[1::2, 1::2, int(num[3])] = 1

        # Generate mosaic
        mosaic = rgb * mask

        return mosaic, mask

    def flatten_to_cfa(self, mosaic_img):
        """
        Converts a 3D mosaic image (output of mosaic function) into a 2D Bayer CFA.
        It uses the GRBG pattern from the mosaic image to create a single-channel CFA.
        """
        h, w, _ = mosaic_img.shape
        cfa = np.zeros((h, w), dtype=mosaic_img.dtype)

        # GRBG Pattern: Flatten the 3D mosaic to 2D
        cfa[::2, ::2] = mosaic_img[::2, ::2, 1]  # Green at (0, 0)
        cfa[::2, 1::2] = mosaic_img[::2, 1::2, 0]  # Red at (0, 1)
        cfa[1::2, ::2] = mosaic_img[1::2, ::2, 2]  # Blue at (1, 0)
        cfa[1::2, 1::2] = mosaic_img[1::2, 1::2, 1]  # Green at (1, 1)

        return cfa

    def load_demosaic_method(self, method_name):
        """
        Dynamically loads the demosaicking method script from the respective folder inside the Demosaicker directory.
        Returns the `demosaic_function` from the script.
        """
        method_folder = os.path.join(self.demosaicker_folder, method_name)
        method_script = f"run_{method_name}.py"
        script_path = os.path.join(method_folder, method_script)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Demosaicking method script not found: {script_path}")

        # Add the method's folder to sys.path so that it can find its dependencies
        sys.path.insert(0, method_folder)

        # Load the script dynamically
        spec = importlib.util.spec_from_file_location(f"run_{method_name}", script_path)
        demosaic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demosaic_module)

        # Check if the loaded module has the necessary `demosaic_function`
        if not hasattr(demosaic_module, 'demosaic_function'):
            raise AttributeError(f"No `demosaic_function` found in {script_path}")
        
        return demosaic_module.demosaic_function

    def psnr(self, gt_img, demosaicked_img):
        """
        Calculate PSNR (r, g, b, all) between ground truth and demosaicked images.
        """
        print(gt_img.dtype, demosaicked_img.dtype)
        psnr_r = cv2.PSNR(gt_img[:, :, 0], demosaicked_img[:, :, 0])
        psnr_g = cv2.PSNR(gt_img[:, :, 1], demosaicked_img[:, :, 1])
        psnr_b = cv2.PSNR(gt_img[:, :, 2], demosaicked_img[:, :, 2])
        psnr_all = (psnr_r + psnr_g + psnr_b) / 3
        
        return psnr_r, psnr_g, psnr_b, psnr_all

    def calculate_ssim(self, gt_img, demosaicked_img):
        """
        Calculate SSIM between ground truth and demosaicked images.
        """
        ssim_value, _ = ssim(gt_img, demosaicked_img, multichannel=True, full=True)
        return ssim_value

    def process_single_image(self, img_path, demosaic_method='GBTf'):
        """
        Processes a single image: applies mosaic, dynamically loads and runs the specified demosaicking method,
        evaluates PSNR and SSIM.
        """
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_name}")
            return
        
        # Mosaic the image
        mosaic_img,mask = self.mosaic_bayer(img,'grbg')
        
        # Convert to CFA
        cfa_img = self.flatten_to_cfa(mosaic_img)

        # Load and apply the demosaicking method
        demosaic_function = self.load_demosaic_method(demosaic_method)
        demosaicked_img = demosaic_function((mosaic_img,mask, self.bayer_type))  # Call the dynamically loaded demosaic function
        
        # Save the demosaicked image
        result_path = os.path.join(self.result_folder, img_name)
        cv2.imwrite(result_path, demosaicked_img)
        print(f"Processed and saved: {result_path}")
        
        # Evaluate PSNR and SSIM
        psnr_r, psnr_g, psnr_b, psnr_all = self.psnr(img, demosaicked_img)
        ssim_value = self.calculate_ssim(img, demosaicked_img)
        
        return psnr_r, psnr_g, psnr_b, psnr_all, ssim_value

    def process_images(self, demosaic_method='GBTF'):
        """
        Processes all images in the dataset folder using the specified demosaicking method.
        Calls process_single_image for each image and logs the results in a CSV file.
        """
        gt_images = os.listdir(self.input_folder)
        csv_file_path = os.path.join(self.result_folder, "results.csv")
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "PSNR_R", "PSNR_G", "PSNR_B", "PSNR_All", "SSIM"])
            
            for img_name in gt_images:
                img_path = os.path.join(self.input_folder, img_name)
                psnr_r, psnr_g, psnr_b, psnr_all, ssim_value = self.process_single_image(img_path, demosaic_method=demosaic_method)
                
                # Write results to CSV
                writer.writerow([img_name, psnr_r, psnr_g, psnr_b, psnr_all, ssim_value])

        print(f"Results saved to {csv_file_path}")
