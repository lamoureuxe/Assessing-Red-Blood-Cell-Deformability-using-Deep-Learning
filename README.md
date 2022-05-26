# Assessing Red Blood Cell Deformability using Deep Learning
Code from:
Lamoureux, E. S., Islamzada, E., Wiens, M. V., Matthews, K., Duffy, S. P., & Ma, H. (2022). Assessing red blood cell deformability from microscopy images using deep learning. Lab on a Chip, 22(1), 26-39. https://doi.org/10.1039/D1LC01006A.

This code implements image segmentation, augmentation, and classification to assess red blood cell deformability. 

1. image_segmentation.py is used to segment single cell 60x60 pixel images from 40X magnification microscope image scans.
2. image_augmentation.py is used to augment the segmented single cell images by random integer multiples of 90 degree rotation to create balanced training and testing classes and to smooth potential bias from imaging well location-based shading bias.
3. main.py uses the balanced augmented classes to process the datasets, process the images, 
