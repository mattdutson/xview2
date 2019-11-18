import os
import random

import numpy as np
import tensorflow as tf


def preprocess_dataset(directory, shuffle=True):
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
    if shuffle:
        random.shuffle(dataset)
    return dataset


def load_image(path):
    return tf.cast(tf.io.decode_png(tf.io.read_file(path)), tf.float32) / 255.0


def generator(dataset, directory, reshuffle=True):
    while True:
        if reshuffle:
            random.shuffle(dataset)

        for item in dataset:
            pre = load_image(os.path.join(directory, "images", item[0]))
            post = load_image(os.path.join(directory, "images", item[1]))

            pre_post = tf.concat([pre, post], axis=-1)
            pre_post = tf.reshape(pre_post, [1, pre_post.shape[0], pre_post.shape[1], pre_post.shape[2]])

            mask = np.load(os.path.join(directory, "raster_labels", item[2]))
            mask = np.reshape(mask, [1, mask.shape[0], mask.shape[1], 1])
            mask = tf.keras.utils.to_categorical(mask, num_classes=5)

            yield pre_post, mask


def compute_class_weights(train, directory, n_classes=5):
    frequencies = np.zeros(n_classes, dtype=np.float32)

    for item in train:
        raster_np = np.load(os.path.join(directory, "raster_labels", item[2]))
        for i in range(n_classes):
            frequencies[i] += np.count_nonzero(raster_np == i)

    weights = frequencies ** -1
    return weights / np.sum(weights)
