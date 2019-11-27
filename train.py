#!/usr/bin/env python3

import argparse

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *

from data_generator import DataGenerator
from unet import create_model
from util import *


def train(args):
    size = (args.x_size, args.y_size)
    if args.crop_x is not None and args.crop_y is not None:
        crop_size = (args.crop_x, args.crop_y)
    else:
        crop_size = None
    train_gen = DataGenerator(args.train_dir, size=size, crop_size=crop_size, shuffle=True, seed=1)
    val_gen = DataGenerator(args.val_dir, size=size, crop_size=crop_size, shuffle=False)

    if crop_size is not None:
        model = create_model(size=crop_size)
    else:
        model = create_model(size=size)
    if args.load is not None:
        model.load_weights(args.load)

    schedule = PiecewiseConstantDecay([2 * len(train_gen)], [0.00001, 0.000001])
    optimizer = RMSprop(learning_rate=schedule)
    loss = WeightedCrossEntropy(np.array([0.03, 1.0, 1.0, 1.0, 1.0]))
    metrics = ["acc"]
    for i in range(5):
        metrics.append(Precision(class_id=i, name="p_{}".format(i)))
        metrics.append(Recall(class_id=i, name="r_{}".format(i)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [PrintXViewMetrics()]
    if args.checkpoint_dir is not None:
        path = os.path.join(args.checkpoint_dir, "checkpoint_{epoch}.h5")
        callbacks.append(ModelCheckpoint(path, save_weights_only=True))
    if args.best is not None:
        callbacks.append(ModelCheckpoint(args.best, monitor="val_loss", save_best_only=True, save_weights_only=True))
    if args.output_dir is not None:
        callbacks.append(SaveOutput(train_gen, args.output_dir, n_items=50))

    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        epochs=args.epochs,
        callbacks=callbacks)

    if args.save is not None:
        model.save_weights(args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b", "--best", default=None, type=str,
        help="path for storing the model with the lowest validation loss")
    parser.add_argument(
        "-c", "--checkpoint_dir", default=None, type=str,
        help="folder for saving model checkpoints after each epoch")
    parser.add_argument(
        "--crop_x", default=None, type=int,
        help="width of random crops")
    parser.add_argument(
        "--crop_y", default=None, type=int,
        help="height of random crops")
    parser.add_argument(
        "-e", "--epochs", default=50, type=int,
        help="number of training epochs")
    parser.add_argument(
        "-l", "--load", default=None, type=str,
        help="path for loading an existing model")
    parser.add_argument(
        "-o", "--output_dir", default=None, type=str,
        help="path for saving sample outputs")
    parser.add_argument(
        "-s", "--save", default=None, type=str,
        help="path for saving the final model")
    parser.add_argument(
        "-t", "--train_dir", default=os.path.join("dataset", "train"), type=str,
        help="folder containing training data")
    parser.add_argument(
        "-v", "--val_dir", default=os.path.join("dataset", "val"), type=str,
        help="folder containing validation data")
    parser.add_argument(
        "--x_size", default=1024, type=int,
        help="width of the model input")
    parser.add_argument(
        "--y_size", default=1024, type=int,
        help="height of the model input")

    train(parser.parse_args())
