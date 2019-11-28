import argparse
import os
import tensorflow as tf

from unet import create_model
from data_generator import TestDataGenerator
from util import write_png

def test():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # num of test cases
    num_test_imgs = 932

    # generate the test dataset
    test_dataset = TestDataGenerator(directory=test_dir)
    
    for i in range(num_test_imgs + 1):
        # get prediction
        pred = model.predict(test_dataset[i])[0, :, :, :]

        # encode to png and write to file
        pred = tf.argmax(pred, axis=-1)
        pred = 50 * tf.cast(pred, tf.uint8)
        pred = tf.expand_dims(pred, axis=-1)
        write_png(pred, os.path.join(output_dir, "test_damage_{:05d}_prediction.png".format(i)))
        write_png(pred, os.path.join(output_dir, "test_localization_{:05d}_prediction.png".format(i)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.json')
    parser.add_argument('--output', type=str, default='predictions')
    args = parser.parse_args()
    
    # test and output directories
    test_dir = os.path.join("dataset", "test")
    output_dir = args.output

    # load the model structure
    model = create_model()

    # load model weights
    model.load_weights(args.model)

    test()
    

