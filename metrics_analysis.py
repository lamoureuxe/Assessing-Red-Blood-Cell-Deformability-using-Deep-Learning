# This function conducts the metrics analysis for our model. It plots the training and validation accuracies and losses
# over the training epochs. Additionally, it evaluates the model using a reciever operating characteristic (ROC) curve.

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import matplotlib.pyplot as plt

def metrics_analysis(history, y_cat_test, y_test_predict):
    print(metrics.classification_report(y_cat_test.argmax(axis=-1), y_test_predict.argmax(axis=-1), digits=3))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')\
    plt.legend(['train'], loc='upper left')
    plt.show()

    # roc curve and auc
    # calculate roc curve
    y_test_predict_2 = y_test_predict[:, 1]
    fpr, tpr, thresholds = roc_curve(y_cat_test.argmax(axis=-1), y_test_predict_2)

    # calculate AUC
    auc = roc_auc_score(y_cat_test, y_test_predict)
    print('AUC: %.3f' % auc)

    pyplot.plot(fpr, tpr, linestyle='-')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()