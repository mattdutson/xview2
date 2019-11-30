#!/usr/bin/env python3

import argparse

from data_generator import TestDataGenerator
from unet import create_model
from util import *


def test(args):
    model = create_model()
    model.load_weights(args.model)

    if not os.path.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)

    test_gen = TestDataGenerator(args.test_dir)

    progress = 0.0
    progress_step = 100.0 / len(test_gen)
    print("Performing test inference...")
    print("Progress: {:3.1f}%\r".format(progress), end="")

    for i in range(len(test_gen)):
        pre_post, index = test_gen[i]

        pred = model.predict(pre_post)[0, :, :, :]
        pred = tf.argmax(pred, axis=-1)
        pred = tf.cast(pred, tf.uint8)
        pred = tf.expand_dims(pred, axis=-1)

        write_png(pred, os.path.join(args.prediction_dir, "test_damage_{:05d}_prediction.png".format(index)))
        write_png(pred, os.path.join(args.prediction_dir, "test_localization_{:05d}_prediction.png".format(index)))

        progress += progress_step
        print("Progress: {:3.1f}%\r".format(progress), end="")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model", type=str, default="model.json",
        help="path for loading model weights")
    parser.add_argument(
        "-o", "--prediction_dir", type=str, default="predictions",
        help="path for saving predictions")
    parser.add_argument(
        "-t", "--test_dir", default=os.path.join("dataset", "test"),
        help="folder containing test data")

    test(parser.parse_args())
