# This file segments 2424x2424 pixel microscope image scans into 60x60 pixel single cell images.

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage.filters import threshold_multiotsu
from skimage.filters import threshold_otsu
import random

# Set up files
#dir_path = os.getcwd()
dir_path = 'I:/rbc_deformability_ai/donor_4'

print(dir_path)
folder_path = dir_path + '/image_scans/donor_4/outlet_3_training'
files = os.listdir(folder_path)

# Look at files
for idx, filenames in enumerate(files):
    print(files[idx])

folder_path_f = folder_path + '\\'

im = cv2.imread(folder_path_f + files[0])
plt.imshow(im, cmap='gray', interpolation = 'nearest')

im_a = []
for idx, images in enumerate(files):
    im_a.append(cv2.imread(folder_path_f + files[idx]))

#check shape and convert to grayscale
for i, im in enumerate(im_a):
    if len(im.shape) == 3:
        gray = [0.2989, 0.5870, 0.1140]
        im = np.dot(im[...,:3], gray)
        im_a[i] = im
    print(im.shape)

# histogram equalization
equ_type = 'adaptive' #'flat','adaptive','none'

for i, im in enumerate(im_a):
    im = im.astype('uint8')
    if equ_type == 'flat':
        im = cv2.equalizeHist(im)
    elif equ_type == 'adaptive':
        clahe_size = 8
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(clahe_size,clahe_size))
        cl1 = clahe.apply(im)
    im_a[i] = im
    plt.imshow(im_a[i], cmap='gray')

# Edge detection using the Sobel operator
figure(num=None, figsize=(80, 80), dpi=80, facecolor='w', edgecolor='k')

crops = []
masks = []

for i, im in enumerate(im_a):

    # edge detection approach using 2nd order gradient (sobel)
    im_1b = ndimage.filters.gaussian_filter(im, sigma=1)
    # im_1b = im_1 #in case of no gaussian
    im_1b = (im_1b).astype('float32')
    kern_xx = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
    kern_yy = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
    G_xx = ndimage.convolve(im_1b, kern_xx, mode='reflect')
    G_yy = ndimage.convolve(im_1b, kern_yy, mode='reflect')
    im_1_edge = cv2.multiply(G_xx, G_xx) + cv2.multiply(G_yy, G_yy)
    im_1_edge = cv2.sqrt((im_1_edge).astype('float32'))
    std = im_1_edge.std()
    im_1_edge = im_1_edge.clip(im_1_edge.mean() - 2 * std, im_1_edge.mean() + 2 * std)
    im_1_edge = im_1_edge * 255.0 / im_1_edge.max()
    im_1_edge = (im_1_edge).astype('uint8')

    # multi-thresholding
    # mask2 = cv2.inRange(im_1_edge, 200, 255)
    # mask2 = mask2/255
    thresholds = threshold_multiotsu(im_1_edge)
    regions = np.digitize(im_1_edge, bins=thresholds)
    # view = regions
    mask2 = regions == 2
    view = mask2
    # mask2 = cv2.inRange(im_1_edge, thresholds[0], thresholds[1])
    # mask2 = mask2/255
    closing_kernel_size = 3
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    mask2 = cv2.erode((mask2).astype('uint8'), kernel2, iterations=2)
    labeled, nr_objects = ndimage.label((mask2).astype('uint8'))
    sizes = np.bincount(labeled.flat)
    mask = cv2.inRange(sizes, 100, 2000)  # switched for t5
    mask = mask > 0
    mask[0] = False
    mask2 = mask[labeled]

    # open image to retain circles (and fill if holes in middle)
    mask2 = cv2.dilate((mask2).astype('uint8'), kernel2, iterations=1)
    mask2 = ndimage.morphology.binary_fill_holes(mask2)

    masks.append(mask2)

    labeled, nr_objects = ndimage.label(mask2)
    centers = center_of_mass(mask2, labels=labeled, index=range(1, nr_objects + 1))

    print(len(centers))

    half_window_width = 30
    for idx in range(nr_objects):
        y = (centers[idx][0]).astype('uint32')
        x = (centers[idx][1]).astype('uint32')
        top = y - half_window_width
        if top < 0: top = 0
        bottom = y + half_window_width
        if bottom > (im.shape)[0]: bottom = (im.shape)[0]
        left = x - half_window_width
        if left < 0: left = 0
        right = x + half_window_width
        if right > (im.shape)[1]: right = (im.shape)[1]
        crops.append(im[top:bottom, left:right])

plt.imshow(view, cmap='gray', interpolation='nearest')

# Display random cropped image
index = [1, 2, 3, 4, 5]

for i in index:
    my_crop = random.choice(crops)
    print(np.shape(my_crop))
plt.imshow(my_crop, cmap='gray', interpolation = 'nearest')

# Select or reject cropped images based on cropping quality
closing_kernel_size = 3
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(closing_kernel_size,closing_kernel_size))

selected_crops = []
rejected_crops = []

for idx, im in enumerate(crops):
    # edge detection approach using 2nd order gradient (sobel)
    im_1b = ndimage.filters.gaussian_filter(im, sigma = 1)
    #im_1b = im_1 #in case of no gaussian
    im_1b = (im_1b).astype('float32')
    kern_xx = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]
    kern_yy = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
    G_xx = ndimage.convolve(im_1b, kern_xx, mode='reflect')
    G_yy = ndimage.convolve(im_1b, kern_yy, mode='reflect')
    im_1_edge = cv2.multiply(G_xx, G_xx) + cv2.multiply(G_yy, G_yy)
    im_1_edge = cv2.sqrt((im_1_edge).astype('float32'))
    std = im_1_edge.std()
    im_1_edge = im_1_edge.clip(im_1_edge.mean()-2*std, im_1_edge.mean()+2*std)
    im_1_edge = im_1_edge*255.0/im_1_edge.max()
    im_1_edge = (im_1_edge).astype('uint8')
    # multiotsu threshold
    threshold = threshold_otsu(im_1_edge)
    mask2 = im_1_edge > threshold
    mask2 = cv2.dilate((mask2).astype('uint8'), kernel2, iterations = 2)
    labeled, nr_objects = ndimage.label((mask2).astype('uint8'))
    if labeled.max() == 1:
        selected_crops.append(im)
    else:
        rejected_crops.append(im)
    if idx%100 == 0:
        print(idx)
        print(len(selected_crops))

plt.imshow(im)

# Display a selected cropped image
my_crop = random.choice(selected_crops)
print(np.shape(my_crop))

plt.imshow(my_crop, cmap='gray', interpolation = 'nearest')

# Display a rejected cropped image
my_crop = random.choice(rejected_crops)
print(np.shape(my_crop))

plt.imshow(my_crop, cmap='gray', interpolation = 'nearest')

# Save the selected cropped images
save_folder_1 = dir_path + '\\single_cell_cropped'
if not os.path.exists(save_folder_1):
    os.makedirs(save_folder_1)

save_folder_2 = save_folder_1 + '\\outlet_3_training'
if not os.path.exists(save_folder_2):
    os.makedirs(save_folder_2)

for idx, im in enumerate(selected_crops):
    if (np.shape(im)[0] == 2 * half_window_width) and (np.shape(im)[1] == 2 * half_window_width):
        im = Image.fromarray(im)
        save_directory = save_folder_2 + '\\_' + str(idx) + '.png'
        im.save(save_directory)

# Save the rejected cropped images
save_folder_1 = dir_path + '\\rejected_single_cell_images'
if not os.path.exists(save_folder_1):
    os.makedirs(save_folder_1)

save_folder_2 = save_folder_1 + '\\outlet_3_training'
if not os.path.exists(save_folder_2):
    os.makedirs(save_folder_2)

for idx, im in enumerate(rejected_crops):
    if (np.shape(im)[0] == 2 * half_window_width) and (np.shape(im)[1] == 2 * half_window_width):
        im = Image.fromarray(im)
        save_directory = save_folder_2 + '\\_' + str(idx) + '.png'
        im.save(save_directory)