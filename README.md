# object_detect_darknet
Object detection utilizing Darknet-based object detection models such as YOLOv3.

## Build a custom dataset

## Build a Darknet-based object detection model

(NOTE: the below is based on the 
[Darknet guideline for training for custom object detection](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects))

1. Clone the Darknet repository that contains all the code for compiling and 
running the model:
    ```bash
    $ git clone git@github.com:AlexeyAB/darknet.git
    $ cd darknet
    $ export DARKNET=`pwd`
    ```
    Update the Makefile in the `darknet` repository by changing the following 
    attribute values from 0 to 1:
    ```bash
    GPU=1
    CUDNN=1
    CUDNN_HALF=1
    AVX=1
    OPENMP=1
    ```
    Next build the `darknet` executable:
    ```bash
    $ make
    ```
    At this point, assuming the above build completed without an error, we should 
    have an executable `darknet` file located in the ${DARKNET} directory:
    ```bash
    $ ls -l ${DARKNET}/darknet
    -rwxr-xr-x 1 james james 4370776 Sep 11 11:43 darknet
    ```

2. Download/build a custom weapons dataset which currently contains a two image 
classes, "handgun" and "rifle", with annotations in Darknet format. We'll copy 
all image and annotation files into a directory under the directory that will 
contain the `darknet` executable, and then create the `train.txt` and `valid.txt` 
files that will be used later to designate the training and validation datasets 
used as input for training the model.
    ```bash
    $ git clone git@github.com:SecurityCameraWarehouse/dataset_weapons.git
    $ cd dataset_weapons
    $ pip install -r requirements.txt
    $ export PYTHONPATH=`pwd`
    $ mkdir /home/ubuntu/datasets/weapons
    $ export WEAPONS_DIR=/home/ubuntu/datasets/weapons
    $ python dataset_weapons/__main__.py --out_dir ${WEAPONS_DIR} --format darknet --exclusions exclusions.txt
    $ export IMAGES_RELATIVE_PATH=data/weapons
    $ mkdir ${DARKNET}/${IMAGES_RELATIVE_PATH}
    $ export TRAIN_VALID_IMAGES_DIR=${DARKNET}/${IMAGES_RELATIVE_PATH}
    $ cp ${WEAPONS_DIR}/images/*/*.jpg ${TRAIN_VALID_IMAGES_DIR}
    $ cp ${WEAPONS_DIR}/annotations/darknet/*/*.txt ${TRAIN_VALID_IMAGES_DIR}
    $ python darknet_train_valid.py --darknet_dir ${DARKNET} --images_dir ${IMAGES_RELATIVE_PATH} --train_dir ${WEAPONS_DIR}/images/train --valid_dir ${WEAPONS_DIR}/images/val --dest_dir ${DARKNET}/build/darknet/x64/data
    ``` 

3. Create a new `*.cfg` file by copying the configuration file for the model we'll 
use -- in this case we'll copy the configuration file for the "tiny" YOLOv3 model 
for later modification for training with our custom dataset: 
    ```bash
    $ cd ${DARKNET}/cfg
    $ cp yolov3-tiny.cfg yolov3-tiny-weapons-608.cfg
    ```
    The original YOLOv3-tiny model was trained with 80 classes of objects and in 
    our case we'll only have 2 classes (handgun and rifle), so we'll update 
    various entries to account for this. Also the minimum number of iterations 
    (or batches) is advised to be 2000 per class, but it's a good idea to run many 
    more batches in order to achieve optimal loss/accuracy. We'll use 10000 per 
    class in the example below: 
   
    * change the resolution from 416 x 416 to 608 x 608, i.e. `width=608` and 
    `height=608` 
    * change `classes=80` to `classes=2` in each of the two [yolo]-layers
    * change `max_batches` to (classes * 10000), i.e. `max_batches=20000`
    * change `steps` to 80% and 90% of `max_batches`, i.e. `steps=16000,18000`
    * change `filters=255` to `filters=21` in the two [convolutional] layers 
    where this `filters` attribute is present (NOTE: the `filters` value here 
    should equal (\<number of classes\> + 5) * 3)

4. Create the file `obj.names` in the directory `${DARKNET}/build/darknet/x64/data` with 
one object name per line. Use the same order as is used for the object classes in 
the custom dataset.

    For example if we have two classes in our dataset, "handgun" and "rifle", and 
    these are indexed 0: "handgun", 1: "rifle", then the file would look like so:
    ```
    handgun
    rifle
    ```

5. Create the file `obj.data` in the directory `${DARKNET}/build/darknet/x64/data`, 
containing lines specifying the number of object classes (where `classes` = 
\<number of objects\>), location of the training and validation image list files, 
location of the `obj.names` file we created above, and the directory to be used 
for backup. The directories are relative to the `${DARKNET}` (darknet repository 
home) directory. 
    
    Example:
    ```
    classes=2
    train=build/darknet/x64/data/train.txt
    valid=build/darknet/x64/data/valid.txt
    names=build/darknet/x64/data/obj.names
    backup=backup/
    ```

6. Since the configuration file we're using originally had an input resolution 
(width and height) of 416 x 416 and we've modified our version to 604 x 604 we 
will also need to update the anchors configuration setting. We'll use the `darknet` 
executable to calculate the appropriate anchors for this resolution on our dataset, 
and then copying/pasting the computed anchor values as the `anchors` setting in 
the configuration file `${DARKNET}/cfg/yolov3-tiny-weapons-608.cfg`
    ```bash
    $ ./darknet detector calc_anchors ${DARKNET}/build/darknet/x64/data/obj.data -num_of_clusters 6 -width 608 -height 608 -show
    ```
     
7. Download the default weights for the convolutional layers from 
[here](https://pjreddie.com/media/files/yolov3-tiny.weights) and put into the 
directory `${DARKNET}/build/darknet/x64`:
    ```bash
    $ cd ${DARKNET}/build/darknet/x64
    $ wget https://pjreddie.com/media/files/yolov3-tiny.weights
    ```

8. Get the pre-trained weights `yolov3-tiny.conv.15` that we'll load to 
initialize the model during training:
    ```bash
    $ cd ${DARKNET}
    $ ./darknet partial cfg/yolov3-tiny-weapons-608.cfg build/darknet/x64/yolov3-tiny.weights yolov3-tiny.conv.15 15
    ```
   
9. Train the model on a single GPU:
    ```bash
    $ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-tiny-weapons-608.cfg yolov3-tiny.conv.15
    ```  

    As the model is training it will save the trained weights at every 1000 
    iterations into the `backup` directory specified in the file 
    `${DARKNET}/build/darknet/x64/data/obj.data`. For example after completion of 
    2000 batches:
    ```bash
    $ ls -l ${DARKNET}/backup
    -rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 22:09 yolov3-tiny-weapons-608_1000.weights
    -rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 23:13 yolov3-tiny-weapons-608_2000.weights
    -rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 23:13 yolov3-tiny-weapons-608_final.weights
    -rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 23:13 yolov3-tiny-weapons-608_last.weights
    ```
    _Multi-GPU_ (optional):
    
    If multiple GPUs are available then we can stop the training (after at least 
    1000 iterations) and restart using the latest weights and specifying 
    multiple GPU IDs so as to parallelize the training over multiple (up to 4) GPUs.
    
    The configuration file will need to modified to adjust the learning rate setting 
    to be equal to `0.001 / <number_of_gpus>`. For example, if using 4 GPUs then we'll 
    adjust the value in `${DARKNET}/build/darknet/x64/data/obj.data cfg/yolov3-tiny-weapons-608.cfg`
    to `learning_rate=0.00025`.
    
    Assuming that we'll use 4 GPUs and the GPU IDs we'll want to use on our machine 
    are 0, 1, 2, and 3, then we'll restart the training by using the latest training 
    weights file and specifying the GPU IDs with the `-gpus` option:
    ```bash
    $ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-tiny-weapons-608.cfg backup/yolov3-tiny-weapons-608_2000.weights -gpus 0,1,2,3
    ``` 


## Utilize the trained model for object detection (inferencing)
#### Object detection on image files
Perform object detection on all image files in a directory and display the 
images with labelled bounding boxes:
```bash
$ python3 detect_image.py --images_dir /data/datasets/weapons/test \
    --weights /home/james/darknet/20191004/yolov3-tiny-weapons-416_final.weights \
    --config /home/james/darknet/20191004/yolov3-tiny-weapons-416.cfg \
    --labels /home/james/darknet/20191004/labels.txt \
    --confidence 0.6
```
#### Object detection on video stream
Perform object detection on all frames of a video stream and display the 
video with labelled bounding boxes:
```bash
$ python3 detect_video.py --video_url rtsp://username:password@71.85.124.145/unicast/c2/s1 \
    --weights /home/james/darknet/20191004/yolov3-tiny-weapons-416_final.weights \
    --config /home/james/darknet/20191004/yolov3-tiny-weapons-416.cfg \
    --labels /home/james/darknet/20191004/labels.txt --confidence 0.6
```
#### Record detection annotations
Perform object detection on all image files in a directory and write the detections 
as Darknet format annotation files:
```bash
$ python3 annotate_image.py --images_dir /data/datasets/weapons/images \
    --annotations_dir /data/datasets/weapons/darknet
    --weights /home/james/darknet/yolov3-tiny-416.weights \
    --config /home/james/darknet/yolov3-tiny-416.cfg \
    --labels /home/james/darknet/yolov3/labels.txt \
    --confidence 0.6
```
