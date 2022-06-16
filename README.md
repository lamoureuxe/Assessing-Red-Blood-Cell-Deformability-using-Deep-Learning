# Assessing Red Blood Cell Deformability using Deep Learning
Code from:
Lamoureux, E. S., Islamzada, E., Wiens, M. V., Matthews, K., Duffy, S. P., & Ma, H. (2022). Assessing red blood cell deformability from microscopy images using deep learning. Lab on a Chip, 22(1), 26-39. https://doi.org/10.1039/D1LC01006A.

This code implements image segmentation, augmentation, and classification to assess red blood cell deformability. 

1. image_segmentation.py is used to segment single cell 60x60 pixel images from 40X magnification 2424x2424 pixel microscope image scans.
2. image_augmentation.py is used to augment the segmented single cell images by random integer multiples of 90 degree rotation to create balanced training and testing classes and to smooth potential bias from imaging well location-based shading bias.
3. main.py loads the balanced class augmented images, processes the images, processes the training, validation, and testing datasets, loads the GPU, trains a convolutional neural network with cross-validation, tests the model on the held-out testing set, and evaluates the model using confusion matrices, precision, recall, f1-score, and ROC AUC scores, and saliency maps. 

40X magnification 2424x2424 brightfield microscope image scans used in this work are contained in the FRDR database, accessible here: 	https://doi.org/10.20383/103.0589.
