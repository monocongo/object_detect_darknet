"""
Script to perform object detection on all images in a directory. Images will be
displayed including detection bounding boxes.

Example usage:

$ python3 detect_video.py --video_url rtsp://123.45.67.89:1234 \
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
        "--video_url",
        required=True,
        type=str,
        help="RTSP video URL",
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
        required=False,
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

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    video_stream = cv2.VideoCapture(args["video_url"])

    # loop over each image frame in the video stream
    while True:

        # read the next frame from the file
        (grabbed, frame) = video_stream.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # perform object detection on the image, get the image annotated with bounding boxes
        image = detect_objects(frame, darknet, labels, label_colors, args["confidence"], layer_names)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    # successful completion
    exit(0)
