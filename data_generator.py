import os
import random

import numpy as np
import tensorflow as tf

from util import read_png


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            directory,
            size=(1024, 1024),
            n_classes=5,
            shuffle=True,
            seed=0,
            crop_size=None,
            augment=False,
            brightness_range=0.05,
            contrast_range=0.2,
            saturation_range=0.4,
            hue_range=0.05):
        self.size = size
        self.n_classes = n_classes
        self.crop_size = crop_size
        self.augment = augment
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

        self.image_dir = os.path.join(directory, "images")
        self.mask_dir = os.path.join(directory, "masks")
        self.dataset = []
        image_list = os.listdir(self.image_dir)
        mask_list = os.listdir(self.mask_dir)
        for filename in image_list:
            if "pre" in filename:
                post_filename = filename.replace("pre", "post")
                self.dataset.append((filename, post_filename, post_filename))
                if post_filename not in image_list:
                    raise AssertionError(post_filename + " not found in " + self.image_dir)
                if post_filename not in mask_list:
                    raise AssertionError(post_filename + " not found in " + self.mask_dir)

        if shuffle:
            random.seed(seed)
            random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.augment:
            l_r = np.random.choice([True, False])
            u_d = np.random.choice([True, False])

        def augment(image, spatial_only):
            if l_r:
                image = tf.image.flip_left_right(image)
            if u_d:
                image = tf.image.flip_up_down(image)

            if not spatial_only:
                image = tf.image.random_brightness(image, self.brightness_range)
                image = tf.image.random_contrast(image, 1.0 - self.contrast_range, 1.0 + self.contrast_range)
                image = tf.image.random_saturation(image, 1.0 - self.saturation_range, 1.0 + self.saturation_range)
                image = tf.image.random_hue(image, self.hue_range)

            return image

        item = self.dataset[index % len(self.dataset)]
        pre = read_png(os.path.join(self.image_dir, item[0]))
        post = read_png(os.path.join(self.image_dir, item[1]))

        if self.augment:
            pre = augment(pre, False)
            post = augment(post, False)

        pre_post = tf.concat([pre, post], axis=-1)
        pre_post = tf.cast(pre_post, tf.float32) / 255.0
        pre_post = tf.image.resize(pre_post, self.size)
        pre_post = tf.expand_dims(pre_post, axis=0)

        mask = read_png(os.path.join(self.mask_dir, item[2]))

        if self.augment:
            mask = augment(mask, True)

        mask = tf.image.resize(mask, self.size, method="nearest")
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.keras.utils.to_categorical(mask, num_classes=self.n_classes)

        if self.crop_size is not None:
            w = self.crop_size[0]
            h = self.crop_size[1]
            x = np.random.randint(0, pre_post.shape[1] - w)
            y = np.random.randint(0, pre_post.shape[2] - h)
            return pre_post[:, x:x + w, y:y + h, :], mask[:, x:x + w, y:y + h, :]
        else:
            return pre_post, mask

    def class_weights(self, beta=None):
        frequencies = np.zeros(self.n_classes, dtype=np.float32)
        for item in self.dataset:
            mask = read_png(os.path.join(self.mask_dir, item[2]))
            for i in range(self.n_classes):
                frequencies[i] += np.count_nonzero(mask == i)

        if beta is None:
            weights = frequencies ** -1
        else:
            frequencies /= len(self)
            weights = (1.0 - beta) / (1.0 - beta ** frequencies)
        weights /= np.mean(weights)
        return weights


class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, size=(1024, 1024)):
        self.size = size

        self.image_dir = os.path.join(directory, "images")
        self.dataset = []
        image_list = os.listdir(self.image_dir)
        for filename in image_list:
            if "pre" in filename:
                post_filename = filename.replace("pre", "post")
                index = int(filename.replace("test_pre_", "").replace(".png", ""))
                self.dataset.append((filename, post_filename, index))
                if post_filename not in image_list:
                    raise AssertionError(post_filename + " not found in " + self.image_dir)

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

        return pre_post, item[2]
