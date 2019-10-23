"""
Script to perform object detection on all images in a directory. Images will be
displayed including detection bounding boxes.

Example usage:

$ python3 detect_image.py --images_dir /data/datasets/weapons/test \
    --weights /home/james/darknet/20191004/yolov3-tiny-weapons-416_final.weights \
    --config /home/james/darknet/20191004/yolov3-tiny-weapons-416.cfg \
    --labels /home/james/darknet/20191004/labels.txt \
    --confidence 0.6
"""

import argparse
import os

import cv2
import numpy as np

from object_detect_darknet.detect import detect_objects
from object_detect_darknet.utils import resize_image


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="Directory containing one or more images to be used as input",
    )
    args_parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to Darknet model weights file",
    )
    args_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to Darknet model configuration file",
    )
    args_parser.add_argument(
        "--labels",
        required=True,
        type=str,
        help="Path to file specifying the labels used for Darknet model training",
    )
    args_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum probability required for detections",
    )
    args = vars(args_parser.parse_args())

    # load the model from weights/configuration
    darknet = cv2.dnn.readNetFromDarknet(args["config"], args["weights"])

    # get the output layer names
    layer_names = darknet.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in darknet.getUnconnectedOutLayers()]

    # read labels into a list, get colors to represent each
    labels = open(args["labels"]).read().strip().split("\n")
    np.random.seed(42)
    label_colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # loop over each image file in the specified images directory
    image_file_names = os.listdir(args["images_dir"])
    for image_file_name in image_file_names:

        # read the image data
        image_file_path = os.path.join(args["images_dir"], image_file_name)
        image = cv2.imread(image_file_path)

        # perform object detection on the image, update the image with bounding boxes
        image = detect_objects(image, darknet, labels, label_colors, args["confidence"], layer_names)

        # resize to fit on a screen (for handling especially large images)
        display_width = 800
        display_height = 400
        if (image.shape[0] > display_height) or (image.shape[1] > display_width):
            image = resize_image(image, display_width, display_height)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    # successful completion
    exit(0)
