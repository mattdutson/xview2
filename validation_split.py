#!/usr/bin/env python

import argparse
import os
import random


def ensure_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def validation_split(args):
    train_image_dir = os.path.join(args.train_dir, "images")
    train_label_dir = os.path.join(args.train_dir, "labels")
    train_mask_dir = os.path.join(args.train_dir, "masks")

    val_image_dir = os.path.join(args.val_dir, "images")
    val_label_dir = os.path.join(args.val_dir, "labels")
    val_mask_dir = os.path.join(args.val_dir, "masks")

    ensure_exists(args.val_dir)
    ensure_exists(val_image_dir)
    ensure_exists(val_label_dir)
    ensure_exists(val_mask_dir)

    pre_images = []
    for filename in os.listdir(train_image_dir):
        if "pre" in filename:
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
        base_post = base_pre.replace("pre", "post")

        # Move pairs of images
        os.rename(
            os.path.join(train_image_dir, base_pre) + ".png",
            os.path.join(val_image_dir, base_pre) + ".png")
        os.rename(
            os.path.join(train_image_dir, base_post) + ".png",
            os.path.join(val_image_dir, base_post) + ".png")

        # Move pairs of labels
        os.rename(
            os.path.join(train_label_dir, base_pre) + ".json",
            os.path.join(val_label_dir, base_pre) + ".json")
        os.rename(
            os.path.join(train_label_dir, base_post) + ".json",
            os.path.join(val_label_dir, base_post) + ".json")

        # Move pairs of rasterized labels
        os.rename(
            os.path.join(train_mask_dir, base_pre) + ".png",
            os.path.join(val_mask_dir, base_pre) + ".png")
        os.rename(
            os.path.join(train_mask_dir, base_post) + ".png",
            os.path.join(val_mask_dir, base_post) + ".png")

        progress += progress_step
        print("Progress: {:3.1f}%\r".format(progress), end="")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fraction", default=0.1, type=float,
        help="fraction of training items to hold out for validation")
    parser.add_argument(
        "--seed", default=0, type=int,
        help="random seed")
    parser.add_argument(
        "--train_dir", default=os.path.join("dataset", "train"), type=str,
        help="folder containing training data")
    parser.add_argument(
        "--val_dir", default=os.path.join("dataset", "val"), type=str,
        help="folder for validation data")

    validation_split(parser.parse_args())
