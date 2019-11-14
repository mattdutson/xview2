#!/usr/bin/env python3

import unet
from data_generator import generator

if __name__ == "__main__":
    
    # Set up dataloader
    #   Read data from disk
    #   Data augmentation
    #   One-hot encode labels
    
    # Compute class weights (we want a weighted loss function)
    train_gen = generator("data/train")
    val_gen = generator("data/val")

    model = unet.create_model()

    model.fit_generator(generator=train_gen, validation_data=val_gen)