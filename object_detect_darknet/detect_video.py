"""
Script to perform object detection on all images in a directory. Images will be
displayed including detection bounding boxes.

Example usage:

$ python3 detect_video.py --video_url rtsp://username:passwd@71.85.154.125/unicast/c2/s1 \
    --weights /home/james/darknet/20191004/yolov3-tiny-weapons-416_final.weights \
    --config /home/james/darknet/20191004/yolov3-tiny-weapons-416.cfg \
    --labels /home/james/darknet/20191004/labels.txt \
    --confidence 0.6
"""

import argparse
import time

import cv2
import numpy as np

from object_detect_darknet.detect import detect_objects

_DIFFERENCE_THRESHOLD = 15.0

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

    # initialize the video stream, wait a few seconds to allow
    # the camera sensor to warm up
    video_stream = cv2.VideoCapture(args["video_url"])
    time.sleep(2.0)
    cv2.waitKey(1) & 0xFF

    # get the initial frame, in order to have a baseline image for motion detection
    (grabbed, previous_frame) = video_stream.read()

    # loop over each image frame in the video stream
    while True:

        # read the next frame from the file
        (grabbed, frame) = video_stream.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # only perform detection if we've read a significantly different frame
        difference = np.sum(np.absolute(frame - previous_frame)) / np.size(frame)
        if difference > _DIFFERENCE_THRESHOLD:

            # update the previous frame for the next iteration
            previous_frame = frame

            # perform object detection on the image, get the image annotated with bounding boxes
            frame = detect_objects(frame, darknet, labels, label_colors, args["confidence"], layer_names)

        # display the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # successful completion
    exit(0)
