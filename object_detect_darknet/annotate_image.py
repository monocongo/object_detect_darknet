"""
Script to perform object detection on all images in a directory and write . Images will be
displayed including detection bounding boxes.

Example usage:

$ python3 annotate_image.py --images_dir /data/imgs/fedex \
    --annotations_dir /data/imgs/fedex/darknet \
    --weights /home/james/darknet/20191004/yolov3-tiny-weapons-416_final.weights \
    --config /home/james/darknet/20191004/yolov3-tiny-weapons-416.cfg \
    --labels /home/james/darknet/20191004/labels.txt \
    --confidence 0.6
"""

import argparse
import os
import shutil

import cv2

from object_detect_darknet.detect import bounding_boxes


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
        "--annotations_dir",
        required=True,
        type=str,
        help="Directory where Darknet annotation files will be written",
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
        help="Path to file specifying the labels used for Darknet model "
             "training, will be copied into the annotations directory",
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

    # loop over each image file in the specified images directory
    image_file_names = os.listdir(args["images_dir"])
    for image_file_name in image_file_names:

        # skip non-image files, directories, etc.
        if os.path.splitext(image_file_name)[1].lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # read the image data
        image_file_path = os.path.join(args["images_dir"], image_file_name)
        image = cv2.imread(image_file_path)

        # TODO read the model configuration to get the expected image resolution
        model_input_resolution = (416, 416)

        boxes = bounding_boxes(image, darknet, args["confidence"], model_input_resolution)
        if len(boxes) > 0:
            file_id = os.path.splitext(image_file_name)[0]
            darknet_file_path = os.path.join(args["annotations_dir"], file_id + ".txt")
            with open(darknet_file_path, "w") as darknet_file:
                for box in boxes:
                    darknet_file.write(
                        f"{box.class_id} {box.center_x:.4f} {box.center_y:.4f} "
                        f"{box.width:.4f} {box.height:.4f}\n",
                    )

    # copy the labels file into the annotations directory so we'll
    # have a way to cross-reference what each class ID represents
    shutil.copy2(args["labels"], args["annotations_dir"])

    # successful completion
    exit(0)
