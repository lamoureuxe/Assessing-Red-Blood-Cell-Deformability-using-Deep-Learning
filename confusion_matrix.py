# This function implements a confusion matrix to assess the TP/FP/TN/FN rates of the model for training and testing.

import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(model, x_train, y_train, y_cat_train, x_test, y_test, y_cat_test):
    from sklearn.metrics import confusion_matrix
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    from sklearn.utils.multiclass import unique_labels

    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    # Plot training confusion matrix
    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(y_cat_train.argmax(axis=-1), y_train_predict.argmax(axis=-1),
                                     classes=np.unique(y_train), cmap=plt.cm.Blues, normalize=normalize)

    plt.show()

    # Plot testing confusion matrix
    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(y_cat_test.argmax(axis=-1), y_test_predict.argmax(axis=-1),
                                     classes=np.unique(y_test), cmap=plt.cm.Blues, normalize=normalize)

    plt.show()

    return y_test_predict
