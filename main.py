# Image Classification of Red Blood Cell Deformability
# Matthew Wiens (@MatthewWiens101) & Erik Lamoureux (@lamoureuxe)
# University of British Columbia
# December 2021

# Use Case: Donor 4 Training and Testing
# Outlets 3 v 4&5
# 40 epochs, lr = 0.001, cross five-fold validation, saliency maps

# Instructions:
# 1. Run image_segmentation.py for each outlet for the specified donor. This function segments each microscope scan
#    image into 60x60 pixel single-cell images
# 2. Run image_augmentation.py for each outlet for the specified donor. This function augments the single cell images
#    creating balanced classes for training and testing
# 3. Run main.py. The balanced class folders from image_augmentation.py are processed and input to the CNN for image
#    classification training, subsequent testing, and evaluation.

from platform import python_version
print('python_version: ', python_version())

# input directory_path and folder_paths for training and testing
from load_images import load_images
dir_path = 'I:/rbc_deformability_ai/donor_4/single_cell_images_augmented'
folder_path_deform_train = '/outlet_3_training'
folder_path_rigid_train = '/outlets_4_5_training'
folder_path_deform_test = '/outlet_3_testing'
folder_path_rigid_test = '/outlets_4_5_testing'
print('directory_path: ', dir_path)
im_train, im_test, labels_train, labels_test = load_images(dir_path, folder_path_deform_train, folder_path_rigid_train,
                                                           folder_path_deform_test, folder_path_rigid_test)

# process images into training and testing sets.
from process_images import process_images
im_train, im_test, labels_train, labels_test = process_images(im_train, im_test, labels_train, labels_test)

# process datasets into training, testing, and validation sets with encoded categorical labels.
from process_datasets import process_datasets
x_train, x_val, x_test, y_train, y_cat_train, y_cat_val, y_test, y_cat_test = process_datasets(im_train, im_test,
                                                                                               labels_train, labels_test)

# load GPU using tensorflow-gpu for increased learning speed
from load_gpu import load_gpu
load_gpu()

# conduct convolutional neural network (CNN) training using cross-fold validation
from cnn_crossfold_val_train import cnn_crossfold_val_train
model, history = cnn_crossfold_val_train(x_train, x_val, y_cat_train, y_cat_val)

# test previously trained CNN
from test_cnn import test_cnn
test_cnn(model, x_test, y_cat_test)

# conduct confusion matrix analysis
from confusion_matrix import confusion_matrix
y_test_predict = confusion_matrix(model, x_train, y_train, y_cat_train, x_test, y_test, y_cat_test)

# conduct additional metrics analysis including precision, recall, f1-score, and ROC AUC
from metrics_analysis import metrics_analysis
metrics_analysis(history, y_cat_test, y_test_predict)

# conduct saliency maps
from saliency_maps import saliency_maps
saliency_maps(model, x_test, y_cat_test)
