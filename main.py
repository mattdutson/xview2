#!/usr/bin/env python3

import os

from tensorflow.keras.models import model_from_json

from data_generator import preprocess_dataset, generator
from unet import create_model, optimizer, loss, metrics


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


if __name__ == "__main__":
    # Create data generators
    train_dir = os.path.join("dataset", "train")
    val_dir = os.path.join("dataset", "val")
    train = preprocess_dataset(train_dir)
    val = preprocess_dataset(val_dir)
    train_gen = generator(train, train_dir)
    val_gen = generator(val, val_dir)

    # Initialize the model
    load_model = False
    if load_model:
        model = load_model_arch()
        model = load_model_weights("weights.h5", model)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        model = create_model()

    # Train the model
    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train),
        validation_steps=len(val),
        epochs=1,
        verbose=2)

    # Save the model
    save_model_arch(model)
    save_model_weights(model)
