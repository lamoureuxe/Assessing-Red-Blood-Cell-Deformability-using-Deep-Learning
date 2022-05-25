import numpy as np
import tensorflow as tf
from tensorflow import keras

# This function implements a convolutional neural network for image classification. The model is trained using a
# five-fold cross validation scheme.

def cnn_crossfold_val_train(x_train, x_val, y_cat_train, y_cat_val):

    # Crossfold validation
    from sklearn.model_selection import KFold

    acc_per_fold = []
    loss_per_fold = []

    # Merge inputs and targets
    inputs = np.concatenate((x_train, x_val), axis=0)
    targets = np.concatenate((y_cat_train, y_cat_val), axis=0)

    kfold = KFold(n_splits=2, shuffle=True)
    fold_num = 1
    for train, val in kfold.split(inputs, targets):
        # Deep Learning Implementation
        model = tf.keras.models.Sequential([
            # 1st Convolutional Layer
            tf.keras.layers.Conv2D(filters=256, kernel_size=(7, 7), padding='valid', activation='relu',
                                   input_shape=(60, 60, 1), strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            # 2nd Convolutional Layerx
            tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='valid', activation='relu', strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            # 3rd Convolutional Layer
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            # 4th Convolutional Layer
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            # 5th Convolutional Layer
            # tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='valid', activation='relu', strides=(1,1)),
            # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
            # tf.keras.layers.BatchNormalization(),
            # Pass to a dense layer
            # tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            # 1st Dense Layer
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            # 2nd Dense Layer
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            # 3rd Dense Layer
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            # Output Layer
            # tf.keras.layers.Dense(2,activation='softmax')])
            tf.keras.layers.Dense(2, activation='softmax', name='visualized_layer')])

        batch_size = 32
        epochs = 10

        loss_fn = tf.keras.losses.CategoricalCrossentropy()

        sgd = keras.optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=loss_fn, optimizer=sgd, metrics=['accuracy'])
        # print(model.summary())

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_num} ...')

        # Fit data to model
        history = model.fit(inputs[train], targets[train],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=True)

        # Generate generalization metrics
        scores = model.evaluate(inputs[val], targets[val], verbose=0)
        print(
            f'Score for fold {fold_num}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_num = fold_num + 1

    print('training evaluation:')
    model.evaluate(x_train, y_cat_train, verbose=1)
    print('validation evaluation:')
    model.evaluate(x_val, y_cat_val, verbose=1)

    return model, history