#!/usr/bin/env python

import argparse
import os
import random


def ensure_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def validation_split(args):
    train_images_dir = os.path.join(args.train_dir, "images")
    train_labels_dir = os.path.join(args.train_dir, "labels")
    train_raster_dir = os.path.join(args.train_dir, "raster_labels")

    val_images_dir = os.path.join(args.val_dir, "images")
    val_labels_dir = os.path.join(args.val_dir, "labels")
    val_raster_dir = os.path.join(args.val_dir, "raster_labels")
    
    ensure_exists(args.val_dir)
    ensure_exists(val_images_dir)
    ensure_exists(val_labels_dir)
    ensure_exists(val_raster_dir)

    images = os.listdir(train_images_dir)
    pre_images = []
    for filename in images:
        if "_pre_" in filename:
            pre_images.append(filename)
    random.seed(args.seed)
    random.shuffle(pre_images)
    n_val = int(args.fraction * len(pre_images)) 

    progress = 0.0
    progress_step = 100.0 / n_val
    print("Moving validation data...")
    print("Progress: {:3.1f}%\r".format(progress), end="")

    for filename in pre_images[0: n_val]:
        base_pre = os.path.basename(filename).split(".")[0]
        base_post = base_pre.replace("_pre_", "_post_")
        
        # Move pairs of images
        os.rename(
            os.path.join(train_images_dir, base_pre) + ".png",
            os.path.join(val_images_dir, base_pre) + ".png")
        os.rename(
            os.path.join(train_images_dir, base_post) + ".png",
            os.path.join(val_images_dir, base_post) + ".png")
        
        # Move pairs of labels
        os.rename(
            os.path.join(train_labels_dir, base_pre) + ".json",
            os.path.join(val_labels_dir, base_pre) + ".json")
        os.rename(
            os.path.join(train_labels_dir, base_post) + ".json",
            os.path.join(val_labels_dir, base_post) + ".json")
        
        # Move pairs of rasterized labels
        os.rename(
            os.path.join(train_raster_dir, base_pre) + ".npy",
            os.path.join(val_raster_dir, base_pre) + ".npy")
        os.rename(
            os.path.join(train_raster_dir, base_post) + ".npy",
            os.path.join(val_raster_dir, base_post) + ".npy")

        progress += progress_step
        print("Progress: {:3.1f}%\r".format(progress), end="")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir")
    parser.add_argument("val_dir")
    parser.add_argument("-f", "--fraction", default=0.1, type=float)
    parser.add_argument("-s", "--seed", default=0, type=int)
    validation_split(parser.parse_args())
