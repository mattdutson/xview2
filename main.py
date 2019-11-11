#!/usr/bin/env python3

import unet

if __name__ == "__main__":
    model = unet.create_model()
    # TODO: Set class_weights argument when calling "model.fit" (use sklearn.utils.class_weight.compute_class_weight)
    # TODO: Model outputs are one-hot encoded (convert greyscale PNG using keras.utils.to_categorical)
