# Augment data by random integer multiples of 90 degree rotation for Donor 4.
# Erik Lamoureux
# May 2021

import os

import random
from skimage import io
from skimage.transform import rotate
from skimage import img_as_ubyte
import cv2

aug_dir_path = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell'
sc_dir_path = 'I:/rbc_deformability_ai/donor_4/single_cell_cropped'

# Augment outlet 3 wells combined for training - 10,000 images

T3_train_aug_path = aug_dir_path + '/outlet_3_training'
T3_train_images = []

for im in os.listdir(sc_dir_path + '/outlet_3_training'):
    T3_train_images.append(os.path.join('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_3_training', im))

images_to_generate = 10000
i = 1

while i <= images_to_generate:
    image = random.choice(T3_train_images)
    original_image = io.imread(image)
    transformed_image = rotate(original_image, angle=random.randint(1, 4) * 90)

    new_image_path = "%s/aug_d4_o3_train_%s.jpg" % (T3_train_aug_path, i)
    transformed_image = img_as_ubyte(
        transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
    cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
    i = i + 1

# Augment outlet 3 held-out set consisting of combined wells for testing - 2,000

T3_test_aug_path = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_3_testing'
T3_test_images = []

for im in os.listdir('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_3_testing'):
    T3_test_images.append(os.path.join('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_3_testing', im))

images_to_generate = 2000
i = 1

while i <= images_to_generate:
    image = random.choice(T3_test_images)
    original_image = io.imread(image)
    transformed_image = rotate(original_image, angle=random.randint(1, 4) * 90)

    new_image_path = "%s/aug_d4_o3_test_%s.jpg" % (T3_test_aug_path, i)
    transformed_image = img_as_ubyte(
        transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
    cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
    i = i + 1

#############################################################

# Augment outlet 4 wells combined for training - 10,000 images

T4_train_aug_path = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_4_training'
T4_train_images = []

for im in os.listdir(
        'I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_4_training'):
    T4_train_images.append(os.path.join('G:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_4_training', im))

images_to_generate = 10000
i = 1

while i <= images_to_generate:
    image = random.choice(T4_train_images)
    original_image = io.imread(image)
    transformed_image = rotate(original_image, angle=random.randint(1, 4) * 90)

    new_image_path = "%s/aug_d4_o4_train_%s.jpg" % (T4_train_aug_path, i)
    transformed_image = img_as_ubyte(
        transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
    cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
    i = i + 1

# Augment outlet 4 held-out set consisting of combined wells for testing - 2,000 images

T4_test_aug_path = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_4_testing'
T4_test_images = []

for im in os.listdir('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_4_testing'):
    T4_test_images.append(os.path.join('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_4_testing', im))

images_to_generate = 2000
i = 1

while i <= images_to_generate:
    image = random.choice(T4_test_images)
    original_image = io.imread(image)
    transformed_image = rotate(original_image, angle=random.randint(1, 4) * 90)

    new_image_path = "%s/aug_d4_o4_test_%s.jpg" % (T4_test_aug_path, i)
    transformed_image = img_as_ubyte(
        transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
    cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
    i = i + 1

#############################################################

# Augment outlet 5 wells combined for training - 10,000 images

T5_train_aug_path = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_5_training'
T5_train_images = []

for im in os.listdir(
        'I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_5_training'):
    T5_train_images.append(os.path.join('G:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_5_training', im))

images_to_generate = 10000
i = 1

while i <= images_to_generate:
    image = random.choice(T5_train_images)
    original_image = io.imread(image)
    transformed_image = rotate(original_image, angle=random.randint(1, 4) * 90)

    new_image_path = "%s/aug_d4_o5_train_%s.jpg" % (T5_train_aug_path, i)
    transformed_image = img_as_ubyte(
        transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
    cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
    i = i + 1

# Augment outlet 5 held-out set consisting of combined wells for testing - 2,000 images

T5_test_aug_path = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_5_testing'
T5_test_images = []

for im in os.listdir('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_5_testing'):
    T5_test_images.append(os.path.join('I:/rbc_deformability_ai/donor_4/single_cell_cropped/outlet_5_testing', im))

images_to_generate = 2000
i = 1

while i <= images_to_generate:
    image = random.choice(T5_test_images)
    original_image = io.imread(image)
    transformed_image = rotate(original_image, angle=random.randint(1, 4) * 90)

    new_image_path = "%s/aug_d4_o5_test_%s.jpg" % (T5_test_aug_path, i)
    transformed_image = img_as_ubyte(
        transformed_image)  # Convert an image to unsigned byte format, with values in [0, 255].
    cv2.imwrite(new_image_path, transformed_image)  # save transformed image to path
    i = i + 1

# Check number of files

folder_path_T3_train = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_3_training'
folder_path_T4_train = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_4_training'
folder_path_T5_train = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_5_training'

files_T3_train = os.listdir(folder_path_T3_train)
files_T4_train = os.listdir(folder_path_T4_train)
files_T5_train = os.listdir(folder_path_T5_train)

print('T3_train: ', len(files_T3_train))
print('T4_train: ', len(files_T4_train))
print('T5_train: ', len(files_T5_train))

folder_path_T3_test = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_3_testing'
folder_path_T4_test = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_4_testing'
folder_path_T5_test = 'I:/rbc_deformability_ai/donor_4/augmented_single_cell/outlet_5_testing'

files_T3_test = os.listdir(folder_path_T3_test)
files_T4_test = os.listdir(folder_path_T4_test)
files_T5_test = os.listdir(folder_path_T5_test)

print('T3_test: ', len(files_T3_test))
print('T4_test: ', len(files_T4_test))
print('T5_test: ', len(files_T5_test))