#!/usr/bin/env python3

import argparse
import os

from tensorflow.keras.models import model_from_json

from data_generator import preprocess_dataset, generator, compute_class_weights
from unet import WeightedCrossEntropy, create_model, optimizer, metrics


def save_model_arch(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


def save_model_weights(model):
    model.save_weights('weights.h5')


def load_model_arch():
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        return model_from_json(loaded_model_json)


def load_model_weights(weight_file, model):
    model.load_weights(weight_file)
    return model


def train(args):
    # Create data generators
    train_dir = os.path.join("dataset", "train")
    val_dir = os.path.join("dataset", "val")
    train_data = preprocess_dataset(train_dir)
    val_data = preprocess_dataset(val_dir)
    train_gen = generator(train_data, train_dir)
    val_gen = generator(val_data, val_dir)

    # Initialize the model
    class_weights = compute_class_weights(train_data, train_dir)
    if args.load:
        model = load_model_arch()
        model = load_model_weights("weights.h5", model)
        model.compile(optimizer=optimizer, loss=WeightedCrossEntropy(class_weights), metrics=metrics)
    else:
        model = create_model(class_weights)

    # Train the model
    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        epochs=args.epochs)

    # Save the model
    save_model_arch(model)
    save_model_weights(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-e", "--epochs", default=50, type=int)
    train(parser.parse_args())
