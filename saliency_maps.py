# This function computes and displays the sample image saliency maps to confirm the model is learning relevant
# morphological features for deformability classification.

import matplotlib.pyplot as plt
import numpy as np
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency

def saliency_maps(model, x_test, y_cat_test):
    # Develop Saliency Maps

    # Image titles
    image_titles = ['image1', 'image2', 'image3']

    # indices_to_visualize = [ 0, 12, 38, 80, 110, 74, 190 ]
    indices_to_visualize = [0, 12, 38]
    input_image = []
    for index_to_visualize in indices_to_visualize:
        # Get input
        input_image.append(x_test[index_to_visualize])
        print(np.shape(input_image))
        # plt.imshow(input_image[index_to_visualize])
    # images = np.asarray([np.array(input_image[0]), np.array(input_image[1]), np.array(input_image[2])])
    # images = np.asarray([np.array(input_image)])
    images = np.asarray(input_image)
    # images=input_image
    print(np.shape(images))

    X = images

    # Rendering
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(images[i], cmap='gray', interpolation='nearest')
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Preparing input data
    # X = preprocess_input(images[...,0])
    print(np.shape(X))
    # X = images

    replace2linear = ReplaceToLinear()

    input_class = y_cat_test[index_to_visualize].argmax(axis=-1)

    score = CategoricalScore(input_class)
    print(score)

    # Create Saliency object.
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)

    # Generate saliency map
    saliency_map = saliency(score, X)

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=16)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(score, X, smooth_samples=20,  # The number of calculating gradients iterations.
                                      smooth_noise=0.20)  # noise spread level.

    # Render
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    for i, title in enumerate(image_titles):
        ax[i].set_title(title, fontsize=14)
        ax[i].imshow(saliency_map[i], cmap='jet')
        ax[i].axis('off')
    plt.tight_layout()
    # plt.savefig('images/smoothgrad.png')
    plt.show()