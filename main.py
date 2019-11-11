#!/usr/bin/env python3

import unet

if __name__ == "__main__":
    
    # Set up dataloader
    #   Read data from disk
    #   Data augmentation
    #   One-hot encode labels
    
    # Compute class weights (we want a weighted loss function)

    model = unet.create_model()
