import tensorflow as tf

from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Input, MaxPooling2D, Softmax
from tensorflow.keras.models import Model

epsilon = 1e-6


def harmonic_mean(items):
    inv_sum = 0.0
    for item in items:
        inv_sum += (item + epsilon) ** -1
    return len(items) / inv_sum


def f1_score(true_positive, pred_positive):
    tp = tf.reduce_sum(tf.cast(tf.logical_and(true_positive, pred_positive), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(true_positive), pred_positive), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(true_positive, tf.logical_not(pred_positive)), tf.float32))

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    return harmonic_mean([p, r])


def loc(y_true, y_pred):
    true_classes = tf.argmax(y_true, axis=-1)
    pred_classes = tf.argmax(y_pred, axis=-1)

    true_building = true_classes > 0
    pred_building = pred_classes > 0
    return f1_score(true_building, pred_building)


def damage(y_true, y_pred):
    true_classes = tf.argmax(y_true, axis=-1)
    pred_classes = tf.argmax(y_pred, axis=-1)

    f1_scores = []
    for i in range(1, y_pred.shape[-1]):
        true_positive = true_classes == i
        pred_positive = pred_classes == i
        f1_scores.append(f1_score(true_positive, pred_positive))
    return harmonic_mean(f1_scores)


def xview2(y_true, y_pred):
    return 0.3 * loc(y_true, y_pred) + 0.7 * damage(y_true, y_pred)


class WeightedCrossEntropy:
    # class_weights is expected to be a Numpy array
    def __init__(self, class_weights):
        self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.stop_gradient(y_true)
        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        weights = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        return weights * losses


optimizer = "rmsprop"
metrics = ["acc", loc, damage, xview2]


def create_model(class_weights, shape=(1024, 1024, 6,), n_classes=5):
    inputs = Input(shape=shape)

    # Begin contractive layers

    conv_1_1 = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    conv_1_2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv_1_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1_2)

    conv_2_1 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool_1)
    conv_2_2 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv_2_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2_2)

    conv_3_1 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool_2)
    conv_3_2 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv_3_1)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3_2)

    conv_4_1 = Conv2D(512, (3, 3), padding="same", activation="relu")(pool_3)
    conv_4_2 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv_4_1)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4_2)

    # Base of the "U"

    conv_5_1 = Conv2D(1024, (3, 3), padding="same", activation="relu")(pool_4)
    conv_5_2 = Conv2D(1024, (3, 3), padding="same", activation="relu")(conv_5_1)

    # Begin expansive layers

    up_conv_6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation="relu")(conv_5_2)
    concat_6 = Concatenate(axis=-1)([up_conv_6, conv_4_2])
    conv_6_1 = Conv2D(512, (3, 3), padding="same", activation="relu")(concat_6)
    conv_6_2 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv_6_1)

    up_conv_7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation="relu")(conv_6_2)
    concat_7 = Concatenate(axis=-1)([up_conv_7, conv_3_2])
    conv_7_1 = Conv2D(256, (3, 3), padding="same", activation="relu")(concat_7)
    conv_7_2 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv_7_1)

    up_conv_8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation="relu")(conv_7_2)
    concat_8 = Concatenate(axis=-1)([up_conv_8, conv_2_2])
    conv_8_1 = Conv2D(128, (3, 3), padding="same", activation="relu")(concat_8)
    conv_8_2 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv_8_1)

    up_conv_9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation="relu")(conv_8_2)
    concat_9 = Concatenate(axis=-1)([up_conv_9, conv_1_2])
    conv_9_1 = Conv2D(64, (3, 3), padding="same", activation="relu")(concat_9)
    conv_9_2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv_9_1)

    outputs = Conv2D(n_classes, (1, 1), padding="same", activation="relu")(conv_9_2)

    # Set up the optimizer and loss
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=WeightedCrossEntropy(class_weights), metrics=metrics)
    return model
