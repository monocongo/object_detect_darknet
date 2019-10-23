from typing import List

import cv2
import numpy as np

# non-maximum suppression threshold
_NMS_THRESHOLD = 0.3


# ------------------------------------------------------------------------------
def detect_objects(
        image: np.ndarray,
        darknet: cv2.dnn_Net,
        class_labels: List[str],
        class_label_colors: List[tuple],
        confidence_threshold: float,
        layer_names: List[str],
):
    """
    Detect objects in an image and return an image object with all detections
    annotated with labelled bounding boxes.

    :param img_file_path:
    :param darknet:
    :param class_labels:
    :param class_label_colors:
    :param confidence_threshold:
    :param layer_names: output layer names that should be used for detections
    :return:
    """

    # read image into BLOB (assuming model resolution is 416x416)
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

    # get the detection details
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            # extract the class ID and confidence
            # of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_threshold:

                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, _NMS_THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in class_label_colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{class_labels[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(
                image,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return image
