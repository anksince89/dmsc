# Green Interpolation Features Extraction

This project is aimed at extracting various features from an image related to green channel interpolation. The features are extracted using a GRBG Bayer pattern, and the results are saved as individual `.bmp` files for further analysis.

## Table of Contents
1. [Introduction](#introduction)
2. [Bayer Pattern](#bayer-pattern)
3. [Feature Extraction Methods](#feature-extraction-methods)
    - [Horizontal Gradient](#horizontal-gradient)
    - [Vertical Gradient](#vertical-gradient)
    - [Diagonal Gradient](#diagonal-gradient)
    - [Texture Descriptor](#texture-descriptor)
    - [Green Channel Variance](#green-channel-variance)
    - [Standard Deviation](#standard-deviation)
    - [Clustering of Similar Intensities](#clustering-of-similar-intensities)
    - [Gradient Magnitude](#gradient-magnitude)
    - [Symmetry Check](#symmetry-check)
    - [Corner Detection](#corner-detection)
4. [Results](#results)
5. [Conclusion](#conclusion)

---

## Introduction

This project explores a range of factors that could influence the green channel interpolation in demosaicking algorithms. Instead of relying on red and blue channel information, we focus on texture, gradient, and structural analysis of the green channel in an image mosaic.

## Bayer Pattern

A Bayer pattern is a color filter array for arranging RGB color filters on a square grid of photosensors. In this project, we focus on the GRBG pattern, where the first row alternates between green and red pixels.

---

## Feature Extraction Methods

### Horizontal Gradient
We compute the horizontal gradient of the green channel using the Sobel operator:

\[
G_x = \frac{\partial G}{\partial x}
\]

This captures intensity transitions in the horizontal direction.

### Vertical Gradient
Similar to the horizontal gradient, the vertical gradient is computed as:

\[
G_y = \frac{\partial G}{\partial y}
\]

### Diagonal Gradient
Diagonal gradients are computed using both Sobel and Prewitt operators:

\[
G_d = Sobel(G) \quad \text{and} \quad G_d' = Prewitt(G)
\]

### Texture Descriptor
The texture descriptor is computed using Local Binary Patterns (LBP), which analyze local textures within the green channel.

### Green Channel Variance
Variance measures how much green pixel values deviate from their average, computed as:

\[
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
\]

### Standard Deviation
Standard deviation represents the spread of pixel values around the mean.

### Clustering of Similar Intensities
Thresholding is applied to group similar intensities, identifying smooth or textured regions.

### Gradient Magnitude
Gradient magnitude combines both horizontal and vertical gradients:

\[
G = \sqrt{G_x^2 + G_y^2}
\]

### Symmetry Check
This method compares the green channel with its horizontally flipped version to identify symmetry in textures.

### Corner Detection
Harris Corner Detection is used to identify corner points where the gradient changes sharply.

---

## Results

The features extracted from the image are saved as `.bmp` files in the `result` folder. Each image is named after the feature it represents.

---

## Conclusion

This project demonstrates how various features related to the green channel can be extracted and used for green interpolation. These features can aid in understanding texture, edges, and smoothness in images, providing insights for further algorithm development.

---

## Citations
- [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator)
- [Local Binary Patterns (LBP)](https://en.wikipedia.org/wiki/Local_binary_patterns)
