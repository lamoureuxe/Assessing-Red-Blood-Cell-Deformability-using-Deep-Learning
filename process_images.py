import numpy as np
import cv2
import matplotlib.pyplot as plt

# This function processes the images by converting them to grayscale and conducting histogram equalization and
# normalization

def process_images(im_train, im_test, labels_train, labels_test):

    # Training
    # check shape and convert to grayscale
    for i, im_a in enumerate(im_train):
        if len(im_a.shape) == 3:
            gray = [0.2989, 0.5870, 0.1140]
            im_a = np.dot(im_a[..., :3], gray)
            im_a = np.reshape(im_a, (60, 60, 1))
            im_train[i] = im_a
    print('training grayscale image size: ', im_a.shape)

    # histogram equalization
    equ_type = 'adaptive'  # 'flat','adaptive','none'

    for i, im_a in enumerate(im_train):
        im_a = im_a.astype('uint8')
        if equ_type == 'flat':
            im_a = cv2.equalizeHist(im_a)
        elif equ_type == 'adaptive':
            clahe_size = 8
            clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(clahe_size, clahe_size))
            cl1 = clahe.apply(im_a)
        im_train[i] = im_a
    plt.imshow(im_train[0][:, :, 0], cmap='gray')
    #print(np.unique(im_train[0]))

    # normalization
    # NOTE: distribution is not great, we might be able to normalize over entire distribution
    for i, im_a in enumerate(im_train):
        im_a = im_a / 255.0
        im_train[i] = im_a

    # Testing
    # check shape and convert to grayscale
    for i, im_b in enumerate(im_test):
        if len(im_a.shape) == 3:
            gray = [0.2989, 0.5870, 0.1140]
            im_b = np.dot(im_b[..., :3], gray)
            im_b = np.reshape(im_b, (60, 60, 1))
            im_test[i] = im_b
    print('testing grayscale image size: ', im_b.shape)

    # histogram equalization
    equ_type = 'adaptive' #'flat','adaptive','none'

    for i, im_b in enumerate(im_test):
        im_b = im_b.astype('uint8')
        if equ_type == 'flat':
            im_b = cv2.equalizeHist(im_b)
        elif equ_type == 'adaptive':
            clahe_size = 8
            clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(clahe_size,clahe_size))
            cl1 = clahe.apply(im_b)
        im_test[i] = im_b
    plt.imshow(im_test[0][:,:,0], cmap='gray')
    #print(np.unique(im_test[0]))

    # normalization
    # NOTE: distribution is not great, we might be able to normalize over entire distribution
    for i, im_b in enumerate(im_test):
        im_b = im_b / 255.0
        im_test[i] = im_b

    return im_train, im_test, labels_train, labels_test
