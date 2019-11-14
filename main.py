#!/usr/bin/env python3

import os

import unet
from data_generator import preprocess_dataset, generator, compute_class_weights

if __name__ == "__main__":
    # Create data generators
    train_dir = os.path.join("dataset", "train")
    val_dir = os.path.join("dataset", "val")
    train = preprocess_dataset(train_dir)
    val = preprocess_dataset(val_dir)
    train_gen = generator(train, train_dir)
    val_gen = generator(val, val_dir)

    # Initialize the model
    model = unet.create_model()

    # Train the model
    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train),
        validation_steps=len(val),
        class_weight=compute_class_weights(train, train_dir))