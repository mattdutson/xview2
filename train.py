#!/usr/bin/env python3

import argparse

from tensorflow.keras.callbacks import ModelCheckpoint

from data_generator import DataGenerator
from unet import create_model
from util import *


def train(args):
    size = (args.x_size, args.y_size)
    train_gen = DataGenerator(args.train_dir, size=size, shuffle=True, seed=1)
    val_gen = DataGenerator(args.val_dir, size=size, shuffle=False)

    model = create_model(size=size)
    if args.load is not None:
        model.load_weights(model)

    optimizer = "rmsprop"
    loss = WeightedCrossEntropy(train_gen.class_weights())
    metrics = ["acc", loc, damage, xview2]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = []
    if args.checkpoint_dir is not None:
        path = os.path.join(args.checkpoint_dir, "checkpoint_{epoch}.h5")
        callbacks.append(ModelCheckpoint(path, save_weights_only=True))
    if args.best is not None:
        callbacks.append(ModelCheckpoint(args.best, save_best_only=True, save_weights_only=True))

    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        epochs=args.epochs,
        callbacks=callbacks)

    if args.output_dir:
        for i in range(10):
            item = train_gen[i]
            save_output(model, i, item, args.output_dir)

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
        "-e", "--epochs", default=50, type=int,
        help="number of training epochs")
    parser.add_argument(
        "-l", "--load", default=None, type=str,
        help="path for loading an existing model")
    parser.add_argument(
        "-o", "--output_dir", default=None, type=str,
        help="path for saving sample outputs")
    parser.add_argument(
        "-s", "--save", default="model.h5", type=str,
        help="path for saving the final model")
    parser.add_argument(
        "-t", "--train_dir", default=os.path.join("dataset", "train"), type=str,
        help="folder containing training data")
    parser.add_argument(
        "-v", "--val_dir", default=os.path.join("dataset", "val"), type=str,
        help="folder containing validation data")
    parser.add_argument(
        "-x", "--x_size", default=1024, type=int,
        help="width of the model input")
    parser.add_argument(
        "-y", "--y_size", default=1024, type=int,
        help="height of the model input")

    train(parser.parse_args())
