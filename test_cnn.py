# The test_cnn function tests the trained CNN model with the testing dataset.

def test_cnn(model, x_test, y_cat_test):
    print('testing evaluation:')
    model.evaluate(x_test, y_cat_test, verbose=1)