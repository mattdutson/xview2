import os

import numpy as np
import tensorflow as tf


def preprocess_dataset(directory):
    images_dir = os.path.join(directory, "images")
    raster_dir = os.path.join(directory, "raster_labels")

    pre_images = []
    post_images = []
    raster_npy = []

    for filename in os.listdir(images_dir):
        if "pre" in filename:
            pre_images.append(filename)
        else:
            post_images.append(filename)

    for filename in os.listdir(raster_dir):
        if "post" in filename:
            raster_npy.append(filename)

    dataset = list(zip(sorted(pre_images), sorted(post_images), sorted(raster_npy)))
    return dataset


def load_image(path):
    return tf.io.decode_png(tf.io.read_file(path))


def generator(dataset, directory):
    while True:
        for item in dataset:
            pre_image = item[0]
            post_image = item[1]
            raster_npy = item[2]

            pre_tensor = load_image(os.path.join(directory, "images", pre_image))
            post_tensor = load_image(os.path.join(directory, "images", post_image))
            pre_post = tf.reshape(
                tf.concat([pre_tensor, post_tensor], axis=-1),
                [1, pre_tensor.shape[0], pre_tensor.shape[1], 6])

            raster_npy = np.reshape(
                np.load(os.path.join(directory, "raster_labels", raster_npy)),
                [1, pre_tensor.shape[0], pre_tensor.shape[1], 1])
            raster_tensor = tf.keras.utils.to_categorical(raster_npy, num_classes=5)

            yield pre_post, raster_tensor


def compute_class_weights(train, directory, n_classes=5):
    frequencies = np.zeros(n_classes, dtype=np.float32)

    for item in train:
        raster_np = np.load(os.path.join(directory, "raster_labels", item[2]))
        for i in range(n_classes):
            frequencies[i] += np.count_nonzero(raster_np == i)

    weights = frequencies ** -1
    return weights / np.sum(weights)