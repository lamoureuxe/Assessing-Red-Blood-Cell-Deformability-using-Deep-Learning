import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# This function processes the datasets including encoding labels, train-validation splitting, and dataset shuffling

def process_datasets(im_train, im_test, labels_train, labels_test):

    x_train = np.array(im_train)
    y_train = np.array(labels_train)
    x_test = np.array(im_test)
    y_test = np.array(labels_test)

    del im_train
    del labels_train
    del im_test
    del labels_test
    print(x_test.shape)
    print(x_train.shape)
    print(y_test.shape)
    print(y_train.shape)

    print("Images shape: ", x_train.shape)
    print("Labels shape: ", y_train.shape)
    print("Pixel values are between %f and %f" % (x_train.min(), x_train.max()))
    print("Labels are: ", (np.unique(y_train)))

    # encode labels
    encoder = LabelEncoder()
    y_enc_train = encoder.fit_transform(y_train)
    y_enc_test = encoder.transform(y_test)
    print("Encoded labels: ", np.unique(y_enc_train))

    y_cat_train = tf.keras.utils.to_categorical(y_enc_train)
    y_cat_test = tf.keras.utils.to_categorical(y_enc_test)

    # Train Validation Split -- hold out 20% of data for final validation evaluation after cross-fold validation
    # training
    x_train, x_val, y_cat_train, y_cat_val = train_test_split(x_train, y_cat_train, test_size=0.2)

    # Shuffle datasets
    from sklearn.utils import shuffle

    x_train, y_cat_train = shuffle(x_train, y_cat_train)
    x_val, y_cat_val = shuffle(x_val, y_cat_val)
    x_test, y_cat_test = shuffle(x_test, y_cat_test)

    # print dataset sizes
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    print(y_cat_train.shape)
    print(y_cat_val.shape)
    print(y_cat_test.shape)

    return x_train, x_val, x_test, y_train, y_cat_train, y_cat_val, y_test, y_cat_test