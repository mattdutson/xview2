import argparse
import os

from tensorflow.keras.models import model_from_json
from data_generator import load_image


def load_model_arch(model_file_name):
    with open(model_file_name, 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
        return model_from_json(loaded_model_json)

def load_model_weights(weight_file_name, model):
    model.load_weights(weight_file)
    return model

def test(model, test_dir):
    images_dir = os.path.join(test_dir, "images")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # go through the test set
    for filename in os.listdir(images_dir):
        image = load_image(os.path.join(images_dir, filename))
        ypred = model.predict(image)
        ## TODO: encode to png, and write to file here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='model.json')
    parser.add_argument('--weight_file', type=str, default='weights.h5')
    args = parser.parse_args()
    
    test_dir = os.path.join("dataset", "test")
    output_dir = os.path.join("dataset", "output")

    # load the model structure
    model = load_model_arch(args.model_file)

    # load model weights
    model = load_model_weights(args.weight_file)


    

