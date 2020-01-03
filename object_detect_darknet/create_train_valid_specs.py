import argparse
import shutil
import os

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Usage:
    $ python create_train_valid_specs.py \
          --train_dir /home/ubuntu/datasets/weapons/images/train \
          --valid_dir /home/ubuntu/datasets/weapons/images/val
          --train_file /home/ubuntu/git/darknet/build/darknet/x64/data/train.txt \
          --valid_file /home/ubuntu/git/darknet/build/darknet/x64/data/valid.txt \
    """

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--train_dir",
        required=True,
        type=str,
        help="path to directory containing the training dataset of image files "
             "(*.jpg) and corresponding annotation files (*.txt)",
    )
    args_parser.add_argument(
        "--valid_dir",
        required=True,
        type=str,
        help="path to directory containing the validation dataset of image files "
             "(*.jpg) and corresponding annotation files (*.txt)",
    )
    args_parser.add_argument(
        "--train_file",
        required=False,
        type=str,
        default="train.txt",
        help="path to the (output) training dataset specification file",
    )
    args_parser.add_argument(
        "--valid_file",
        required=False,
        type=str,
        default="valid.txt",
        help="path to the (output) validation dataset specification file",
    )
    args = vars(args_parser.parse_args())

    # write the train.txt file
    with open(args["train_file"], "w") as train_txt:

        # list the files in the training directory
        for file_name in os.listdir(args["train_dir"]):

            if file_name.endswith(".jpg"):

                # write the relative path of the image to the train.txt file
                file_path = os.path.join(args["train_dir"], file_name)
                train_txt.write(file_path + "\n")

    # write the valid.txt file
    with open(args["valid_file"], "w") as train_txt:

        # list the files in the training directory
        for file_name in os.listdir(args["valid_dir"]):

            if file_name.endswith(".jpg"):

                # write the relative path of the image to the train.txt file
                file_path = os.path.join(args["valid_dir"], file_name)
                train_txt.write(file_path + "\n")
