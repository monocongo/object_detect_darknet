import argparse
import os

import cv2
import numpy as np

# non-maximum suppression threshold
_NMS_THRESHOLD = 0.3

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

        # find any images with unexpected dimensions
        image_file_path = os.path.join(args["images_dir"], image_file_name)

        # read image into BLOB (assuming model resolution is 416x416)
        image = cv2.imread(args["image"])
        (height, width) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255.0,
            (416, 416),
            swapRB=True,
            crop=False,
        )

        # pass the BLOB through the model to get detections
        darknet.setInput(blob)
        layer_outputs = darknet.forward(layer_names)

        # TODO draw detections onto image, display image

    # successful completion
    exit(0)
