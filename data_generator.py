import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_dataset(dir):
    images_dir = os.path.join(dir, "images")
    raster_dir = os.path.join(dir, "raster_labels")

    pre_images = list()
    post_images = list()
    raster_npy = list()

    for filename in os.listdir(images_dir):
        if "pre" in filename:
            pre_images.append(filename)
        else:
            post_images.append(filename)

    for filename in os.listdir(raster_dir):
        if "post" in filename:
            raster_npy.append(filename)

    pre_images = sorted(pre_images)
    post_images = sorted(post_images)
    raster_npy = sorted(raster_npy)

    return pre_images, post_images, raster_npy

def load_image(path):
  # load an image
  img = cv2.imread(path)
  img = img[:, :, ::-1]  # BGR -> RGB
  return img

def generator(dir):
    pre, post, npy = preprocess_dataset(dir)
    dataSetSize = len(pre)
    for imgInd in range(dataSetSize):
        pr = pre[imgInd]
        po = post[imgInd]
        raster = npy[imgInd]
        # load the pre and post image
        pre_tensor = load_image(os.path.join(dir, "images", pr))
        post_tensor = load_image(os.path.join(dir, "images/", po))
        pre_post = np.concatenate((pre_tensor, post_tensor), axis=2)
        # load the rasterized npy file
        raster_labels = np.load(os.path.join(dir, "raster_labels", raster))
        yield (tf.convert_to_tensor(pre_post), tf.convert_to_tensor(raster_labels))