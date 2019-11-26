#!/usr/bin/env python3

import argparse

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from data_generator import DataGenerator
from unet import create_model
from util import *


def train(args):
    size = (args.x_size, args.y_size)
    if args.crop_x_size is not None and args.crop_y_size is not None:
        crop_size = (args.crop_x_size, args.crop_y_size)
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
    loss = WeightedCrossEntropy(train_gen.class_weights())
    metrics = ["acc", loc, damage, xview2]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = []
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
        "--best", default=None, type=str,
        help="path for storing the model with the lowest validation loss")
    parser.add_argument(
        "--checkpoint_dir", default=None, type=str,
        help="folder for saving model checkpoints after each epoch")
    parser.add_argument(
        "--crop_x_size", default=None, type=int,
        help="width of random crops")
    parser.add_argument(
        "--crop_y_size", default=None, type=int,
        help="height of random crops")
    parser.add_argument(
        "--epochs", default=50, type=int,
        help="number of training epochs")
    parser.add_argument(
        "--load", default=None, type=str,
        help="path for loading an existing model")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="path for saving sample outputs")
    parser.add_argument(
        "--save", default=None, type=str,
        help="path for saving the final model")
    parser.add_argument(
        "--train_dir", default=os.path.join("dataset", "train"), type=str,
        help="folder containing training data")
    parser.add_argument(
        "--val_dir", default=os.path.join("dataset", "val"), type=str,
        help="folder containing validation data")
    parser.add_argument(
        "--x_size", default=1024, type=int,
        help="width of the model input")
    parser.add_argument(
        "--y_size", default=1024, type=int,
        help="height of the model input")

    train(parser.parse_args())
