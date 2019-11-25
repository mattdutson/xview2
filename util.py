import os

import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import *


def read_png(filename):
    return tf.image.decode_png(tf.io.read_file(filename))


def write_png(array, filename):
    tf.io.write_file(filename, tf.image.encode_png(array))


def harmonic_mean(items):
    inv_sum = 0.0
    for item in items:
        inv_sum += (item + 1e-6) ** -1
    return len(items) / inv_sum


# Determines the fraction a / (a + b), returns 1.0 if a + b == 0
# a and b are assumed to take values 0.0, 1.0, 2.0, ...
def safe_frac(a, b):
    den = tf.maximum(a + b, tf.constant(1.0))
    num = tf.maximum(a, den - (a + b))
    return num / den


def f1_score(actual_positive, pred_positive):
    tp = tf.reduce_sum(tf.cast(tf.logical_and(actual_positive, pred_positive), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(actual_positive), pred_positive), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(actual_positive, tf.logical_not(pred_positive)), tf.float32))

    p = safe_frac(tp, fp)
    r = safe_frac(tp, fn)
    return harmonic_mean([p, r])


def loc(y_true, y_pred):
    true_classes = tf.argmax(y_true, axis=-1)
    pred_classes = tf.argmax(y_pred, axis=-1)

    actual_building = true_classes > 0
    pred_building = pred_classes > 0
    return f1_score(actual_building, pred_building)


def damage(y_true, y_pred):
    true_classes = tf.argmax(y_true, axis=-1)
    pred_classes = tf.argmax(y_pred, axis=-1)

    f1_scores = []
    for i in range(1, y_pred.shape[-1]):
        actual_positive = true_classes == i
        pred_positive = pred_classes == i
        f1_scores.append(f1_score(actual_positive, pred_positive))
    return harmonic_mean(f1_scores)


def xview2(y_true, y_pred):
    return 0.3 * loc(y_true, y_pred) + 0.7 * damage(y_true, y_pred)


class WeightedCrossEntropy:
    # class_weights should be a Numpy array
    def __init__(self, class_weights):
        self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.stop_gradient(y_true)
        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        weights = tf.reduce_mean(self.class_weights * y_true, axis=-1)
        return weights * losses


class SaveOutput(Callback):
    def __init__(self, gen, output_dir, n_items=50):
        self.gen = gen
        self.output_dir = output_dir
        self.n_items = n_items

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs):
        for i in range(self.n_items):
            item = self.gen[i]
            pre_post = item[0]

            pre = pre_post[0, :, :, :3]
            pre = tf.cast(pre * 255.0, tf.uint8)
            write_png(pre, os.path.join(self.output_dir, "{}_{}_pre.png".format(epoch, i)))

            post = pre_post[0, :, :, 3:]
            post = tf.cast(post * 255.0, tf.uint8)
            write_png(post, os.path.join(self.output_dir, "{}_{}_post.png".format(epoch, i)))

            true = item[1][0]
            true = tf.argmax(true, axis=-1)
            true = 50 * tf.cast(true, tf.uint8)
            true = tf.expand_dims(true, axis=-1)
            write_png(true, os.path.join(self.output_dir, "{}_{}_true.png".format(epoch, i)))

            pred = self.model.predict(pre_post)[0, :, :, :]
            pred = tf.argmax(pred, axis=-1)
            pred = 50 * tf.cast(pred, tf.uint8)
            pred = tf.expand_dims(pred, axis=-1)
            write_png(pred, os.path.join(self.output_dir, "{}_{}_pred.png".format(epoch, i)))
