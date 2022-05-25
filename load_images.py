import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# This function loads and enumerates the files in the dataset

def load_images(dir_path, folder_path_deform_train, folder_path_rigid_train, folder_path_deform_test,
                folder_path_rigid_test):
    # Donor Training
    folder_path_deform_train = dir_path + '/Outlet 3 Training'
    folder_path_rigid_train = dir_path + '/Outlets 4 and 5 Training'

    files_deform_train = os.listdir(folder_path_deform_train)
    files_rigid_train = os.listdir(folder_path_rigid_train)

    print('deform_train: ', len(files_deform_train))
    print('rigid_train: ', len(files_rigid_train))

    # Donor Testing
    folder_path_deform_test = dir_path + '/Outlet 3 Testing'
    folder_path_rigid_test = dir_path + '/Outlets 4 and 5 Testing'

    files_deform_test = os.listdir(folder_path_deform_test)
    files_rigid_test = os.listdir(folder_path_rigid_test)

    print('deform_test: ', len(files_deform_test))
    print('rigid_test: ', len(files_rigid_test))

    # Now, label the images

    # Training
    labels_train = ['deform'] * len(files_deform_train)
    #print(len(labels_train))
    labels_train.extend(['rigid'] * len(files_rigid_train))
    #print(len(labels_train))

    folder_path_deform_train = folder_path_deform_train + '\\'
    folder_path_rigid_train = folder_path_rigid_train + '\\'

    #print(len(files_deform_train))
    #print(len(files_rigid_train))

    im_a = cv2.imread(folder_path_deform_train + files_deform_train[0])
    plt.imshow(im_a, cmap='gray', interpolation='nearest')

    im_train = []

    for idz, images in enumerate(files_deform_train):
        im_train.append(cv2.imread(folder_path_deform_train + files_deform_train[idz]))
    for idz, images in enumerate(files_rigid_train):
        im_train.append(cv2.imread(folder_path_rigid_train + files_rigid_train[idz]))
    print('training images: ', len(im_train), ' of ', np.shape(im_train[0]), ' images')

    # Testing

    labels_test = ['deform'] * len(files_deform_test)
    #print(len(labels_test))
    labels_test.extend(['rigid'] * len(files_rigid_test))
    #print(len(labels_test))

    folder_path_deform_test = folder_path_deform_test + '\\'
    folder_path_rigid_test =  folder_path_rigid_test + '\\'

    #print(len(files_deform_test))
    #print(len(files_rigid_test))

    im_b = cv2.imread(folder_path_deform_test + files_deform_test[0])
    plt.imshow(im_b, cmap='gray', interpolation = 'nearest')

    im_test = []

    for idz, images in enumerate(files_deform_test):
        im_test.append(cv2.imread(folder_path_deform_test + files_deform_test[idz]))
    for idz, images in enumerate(files_rigid_test):
        im_test.append(cv2.imread(folder_path_rigid_test + files_rigid_test[idz]))
    print('testing images: ', len(im_test), ' of ', np.shape(im_test[0]), ' images')

    return im_train, im_test, labels_train, labels_test


