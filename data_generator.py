import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from util import read_png


class DataGenerator(Sequence):
    def __init__(self, directory, size=(1024, 1024), n_classes=5, shuffle=True, seed=0):
        self.size = size
        self.n_classes = n_classes

        pre_images = []
        post_images = []
        self.image_dir = os.path.join(directory, "images")
        for filename in os.listdir(self.image_dir):
            if "pre" in filename:
                pre_images.append(filename)
            else:
                post_images.append(filename)

        masks = []
        self.mask_dir = os.path.join(directory, "masks")
        for filename in os.listdir(self.mask_dir):
            if "post" in filename:
                masks.append(filename)

        self.dataset = list(zip(sorted(pre_images), sorted(post_images), sorted(masks)))
        if shuffle:
            random.seed(seed)
            random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index % len(self.dataset)]

        pre = read_png(os.path.join(self.image_dir, item[0]))
        post = read_png(os.path.join(self.image_dir, item[1]))
        pre_post = tf.concat([pre, post], axis=-1)
        pre_post = tf.cast(pre_post, tf.float32) / 255.0
        pre_post = tf.image.resize(pre_post, self.size)
        pre_post = tf.expand_dims(pre_post, axis=0)

        mask = read_png(os.path.join(self.mask_dir, item[2]))
        mask = tf.image.resize(mask, self.size, method="nearest")
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)

        return pre_post, mask

    def class_weights(self):
        frequencies = np.zeros(self.n_classes, dtype=np.float32)
        for item in self.dataset:
            mask = read_png(os.path.join(self.mask_dir, item[2]))
            for i in range(self.n_classes):
                frequencies[i] += np.count_nonzero(mask == i)

        weights = frequencies ** -1
        weights = weights / np.sum(weights)
        return weights
