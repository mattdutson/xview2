#!/usr/bin/env python

import argparse
import json
import os

import numpy as np
import skimage.draw
import skimage.io

damage_codes = {
    "un-classified": 1,
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}


def rasterize_labels(args):
    image_dir = os.path.join(args.train_dir, "images")
    label_dir = os.path.join(args.train_dir, "labels")
    mask_dir = os.path.join(args.train_dir, "masks")

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    progress = 0.0
    progress_step = 100.0 / len(os.listdir(label_dir))
    print("Rasterizing polygons...")
    print("Progress: {:3.1f}%\r".format(progress), end="")

    for filename in os.listdir(label_dir):
        filename = os.path.join(label_dir, filename)
        with open(filename, "r") as labels_file:
            label_json = json.load(labels_file)

            base = os.path.basename(filename).split(".")[0]
            image = skimage.io.imread(os.path.join(image_dir, base) + ".png")

            output = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
            for building in label_json["features"]["xy"]:
                point_strs = building["wkt"].replace("POLYGON ((", "").replace("))", "").split(",")
                n = len(point_strs)
                polygon = np.empty((n, 2))
                for i in range(n):
                    polygon[i] = np.fromstring(point_strs[i], sep=" ")

                damage = damage_codes[building["properties"].get("subtype", "un-classified")]
                mask = damage * skimage.draw.polygon2mask(output.shape, polygon).astype("uint8").T
                output = np.maximum(output, mask)

            skimage.io.imsave(os.path.join(mask_dir, base) + ".png", output, check_contrast=False)
            progress += progress_step
            print("Progress: {:3.1f}%\r".format(progress), end="")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir", default=os.path.join("dataset", "train"), type=str,
        help="folder containing training data")

    rasterize_labels(parser.parse_args())
