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
    images_dir = os.path.join(args.train_dir, "images")
    labels_dir = os.path.join(args.train_dir, "labels")
    output_dir = os.path.join(args.train_dir, "raster_labels")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    progress = 0.0
    progress_step = 100.0 / len(os.listdir(labels_dir))
    print("Rasterizing polygons...")
    print("Progress: {:3.1f}%\r".format(progress), end="")

    for labels_filename in os.listdir(labels_dir):
        labels_filename = os.path.join(labels_dir, labels_filename)
        with open(labels_filename, "r") as labels_file:
            labels_json = json.load(labels_file)

            base = os.path.basename(labels_filename).split(".")[0]
            image = skimage.io.imread(os.path.join(images_dir, base) + ".png")

            output = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
            for building in labels_json["features"]["xy"]:
                point_strs = building["wkt"].replace("POLYGON ((", "").replace("))", "").split(",")
                n = len(point_strs)
                polygon = np.empty((n, 2))
                for i in range(n):
                    polygon[i] = np.fromstring(point_strs[i], sep=" ")

                damage = damage_codes[building["properties"].get("subtype", "un-classified")]
                mask = damage * skimage.draw.polygon2mask(output.shape, polygon).astype("uint8").T
                output = np.maximum(output, mask)

            np.save(os.path.join(output_dir, base) + ".npy", output)
            progress += progress_step
            print("Progress: {:3.1f}%\r".format(progress), end="")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir")
    rasterize_labels(parser.parse_args())
