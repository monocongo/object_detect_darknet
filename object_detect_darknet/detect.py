from typing import List, Tuple

import cv2
import numpy as np

# non-maximum suppression threshold
_NMS_THRESHOLD = 0.3


# ------------------------------------------------------------------------------
class BoundingBox:

    def __init__(self, confidence: float):
        self.confidence = confidence


# ------------------------------------------------------------------------------
class DarknetBoundingBox(BoundingBox):

    def __init__(
            self,
            class_id: int,
            center_x: float,
            center_y: float,
            width: float,
            height: float,
            confidence: float,
    ):
        BoundingBox.__init__(self, confidence)
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.class_id = class_id


# ------------------------------------------------------------------------------
class KittiBoundingBox(BoundingBox):

    def __init__(
            self,
            label: str,
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int,
            confidence: float,
    ):
        BoundingBox.__init__(self, confidence)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label


# ------------------------------------------------------------------------------
def box_darknet_to_kitti(
        darknet_box: DarknetBoundingBox,
        class_labels: List[str],
        img_width: int,
        img_height: int,
) -> KittiBoundingBox:

    width = darknet_box.width * img_width
    height = darknet_box.height * img_height
    xmin = int((darknet_box.center_x - (darknet_box.width / 2)) * img_width)
    ymin = int((darknet_box.center_y - (darknet_box.height / 2)) * img_height)
    xmax = int(xmin + width)
    ymax = int(ymin + height)

    return KittiBoundingBox(
        class_labels[darknet_box.class_id],
        xmin,
        ymin,
        xmax,
        ymax,
        darknet_box.confidence,
    )


# ------------------------------------------------------------------------------
def bounding_boxes(
        image: np.ndarray,
        darknet: cv2.dnn_Net,
        confidence_threshold: float,
        model_input_resolution: Tuple,
) -> List[DarknetBoundingBox]:
    """
    Detect objects in an image and return all corresponding bounding boxes.

    :param image:
    :param darknet:
    :param confidence_threshold:
    :param model_input_resolution: expected resolution of images when input to
        the Darknet model, for example (416, 416)
    :return: list of DarknetBoundingBox objects
    """

    # get the output layer names that should be used for detections
    layer_names = darknet.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in darknet.getUnconnectedOutLayers()]

    # read image into BLOB
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        model_input_resolution,  # (416, 416),
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
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_threshold:

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append(detection[0:4])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, _NMS_THRESHOLD)

    # ensure at least one detection exists
    bboxes = []
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():

            bounding_box = DarknetBoundingBox(
                class_ids[i],
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
                confidences[i],
            )
            bboxes.append(bounding_box)

    return bboxes


# ------------------------------------------------------------------------------
def annotate(
        image: np.ndarray,
        darknet: cv2.dnn_Net,
        class_labels: List[str],
        class_label_colors: List[Tuple],
        confidence_threshold: float,
        model_input_resolution: Tuple,
) -> np.ndarray:
    """
    Detect objects in an image and return an image object with all detections
    annotated with labelled bounding boxes.

    :param image: numpy array of
    :param darknet:
    :param class_labels:
    :param class_label_colors:
    :param confidence_threshold:
    :param model_input_resolution: expected resolution of images when input to
        the Darknet model, for example (416, 416)
    :return:
    """

    for bbox in bounding_boxes(
            image,
            darknet,
            confidence_threshold,
            model_input_resolution,
    ):
        # convert to a KITTI box which contains pixel X/Y values
        kitti_box = box_darknet_to_kitti(
            darknet_box=bbox,
            class_labels=class_labels,
            img_width=image.shape[1],
            img_height=image.shape[0],
        )

        # draw a bounding box rectangle and a label/confidence text on the image
        color = [int(c) for c in class_label_colors[bbox.class_id]]
        cv2.rectangle(
            img=image,
            pt1=(int(kitti_box.xmin), int(kitti_box.ymin)),
            pt2=(int(kitti_box.xmax), int(kitti_box.ymax)),
            color=color,
            thickness=2,
        )
        cv2.putText(
            img=image,
            text=f"{kitti_box.label}: {bbox.confidence:.2f}",
            org=(kitti_box.xmin + 2, kitti_box.ymin + 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=2,
        )

    return image
