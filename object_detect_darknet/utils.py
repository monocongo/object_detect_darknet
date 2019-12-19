import configparser
from typing import Tuple

import cv2
import numpy as np


# ------------------------------------------------------------------------------
def find_input_resolution(
        config_path: str,
) -> Tuple:
    """
    Read a Darknet configuration file and extract the network's input resolution.

    :param config_path: absolute path to a Darknet configuration file
    :return: (width, height) tuple
    """

    # read the configuration file to find the network's input width and height
    config_parser = configparser.ConfigParser(strict=False)
    model_input_width = 416  # reasonable default
    model_input_height = 416  # reasonable default
    config_parser.read(config_path)
    section = "net"
    if config_parser.has_section(section):
        params = config_parser.items(section)
        for param in params:
            if param[0] == "width":
                model_input_width = int(param[1])
            elif param[0] == "height":
                model_input_height = int(param[1])

    return tuple(model_input_width, model_input_height)


# ------------------------------------------------------------------------------
def resize_image(
        image: np.ndarray,
        new_width: int,
        new_height: int,
) -> (np.ndarray, float, float):
    """
    Reads image data from a file and resizes it to the specified dimensions,
    preserving the aspect ratio and padding on the right and bottom as necessary.

    :param image: image data array
    :param new_width: the new width the image should be resized to
    :param new_height: the new height the image should be resized to
    :return: new image data resized to the new dimensions (may include
        padding to preserve aspect ratio)
    """

    # get the dimensions and aspect ratio
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # determine the interpolation method we'll use
    if (original_height > new_height) or (original_width > new_width):
        # use a shrinking algorithm for interpolation
        interp = cv2.INTER_AREA
    else:
        # use a stretching algorithm for interpolation
        interp = cv2.INTER_CUBIC

    # determine the new width and height (may differ from the width and
    # height arguments if using those doesn't preserve the aspect ratio)
    final_width = new_width
    final_height = round(final_width / aspect_ratio)
    if final_height > new_height:
        final_height = new_height
    final_width = round(final_height * aspect_ratio)

    # at this point we may be off by a few pixels, i.e. over
    # the specified new width or height values, so we'll clip
    # in order not to exceed the specified new dimensions
    final_width = min(final_width, new_width)
    final_height = min(final_height, new_height)

    # get the padding necessary to preserve aspect ratio
    pad_bottom = abs(new_height - final_height)
    pad_right = abs(new_width - final_width)

    # scale and pad the image
    scaled_img = cv2.resize(image, (final_width, final_height), interpolation=interp)
    padded_img = cv2.copyMakeBorder(
        scaled_img, 0, pad_bottom, 0, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )

    return padded_img
