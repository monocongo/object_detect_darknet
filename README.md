# object_detect_darknet
Described below is a recipe for performing object detection utilizing 
[Darknet](https://github.com/AlexeyAB/darknet) -based object detection models such 
as YOLOv3.

(NOTE: the below is based on the 
[Darknet guideline for training for custom object detection](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects))

## Build the Darknet executable


1. Prerequisites
* [CUDA](https://developer.nvidia.com/cuda-toolkit) 
* [CUDA DNN](https://developer.nvidia.com/rdp/cudnn-download)
* OpenCV:
  ```bash
  $ sudo apt-get install libopencv-dev python3-opencv
  ```

2. Clone the Darknet repository that contains all the code for compiling and 
running the model:
    ```bash
    $ git clone git@github.com:AlexeyAB/darknet.git
    $ cd darknet
    $ export DARKNET=`pwd`
    ```
3. Edit the Makefile in the `darknet` repository by changing the following 
attribute values from 0 to 1:
    ```bash
    GPU=1
    CUDNN=1
    CUDNN_HALF=1
    OPENCV=1
    AVX=1
    OPENMP=1
    ```
4. Build the `darknet` executable:
    ```bash
    $ make
    ```
    At this point, assuming the above build completed without an error, we should 
    have an executable `darknet` file located in the ${DARKNET} directory:
    ```bash
    $ ls -l ${DARKNET}/darknet
    -rwxr-xr-x 1 james james 4370776 Sep 11 11:43 darknet
    ```

## Build a custom dataset

Acquire a Darknet-format annotated dataset. Datasets with annotation files in other 
formats can be converted/translated to Darknet format using the Python package 
[cvdata](https://github.com/monocongo/cvdata). As well as the image and corresponding 
annotation files this should also include a labels file that lists one class label 
per line in the order corresponding to the indices used in the Darknet files for 
the class labels. For example, if the Darknet files for an animals dataset uses 
the indices (0: cat, 1: dog, and 2: panda) then the labels file will look like so:
```
cat
dog
panda
```
Once the dataset is available we will perform the following processing steps:

1. Resize the images and associated Darknet format annotation files to the resolution 
required for training input. This should match to the width and height values used 
in the configuration file used for the model being trained.
2. Split the dataset into training and validation subsets. A reasonable example 
split could be 80% for training and 20% for validation. The training and validation 
subsets should be in separate directories in order to facilitate the creation of 
the required image list files in the next step.
3. Create `train.txt` and `valid.txt` files that list the paths to the training 
and validation images. A [utility script](https://github.com/monocongo/object_detect_darknet/object_detect_darknet/create_train_valid_specs.py) 
for this exists in this repository. For example:
```bash
$ python create_train_valid_specs.py --train_dir /data/split_darknet_train \
    --valid_dir /data/split_darknet_valid \
    --dest_dir ${DARKNET}/build/darknet/x64/data/obj \
    --train_file ${DARKNET}/build/darknet/x64/data/train.txt \
    --valid_file ${DARKNET}/build/darknet/x64/data/valid.txt \
```
4. Copy all image and annotation files from the training and validation 
subdirectories into `${DARKNET}/build/darknet/x64/data/obj`

## Model training configuration
1. Create a new `*.cfg` file by copying the configuration file for the model we'll 
use. For example, if using the "tiny" YOLOv3 model then we'll copy the configuration 
file for the "tiny" YOLOv3 model located in the `${DARKNET}/cfg` directory:
for later modification for training with our custom dataset: 
    ```bash
    $ cd ${DARKNET}/cfg
    $ cp yolov3-tiny.cfg yolov3-tiny-weapons-608.cfg
    ```
2. The original YOLOv3-tiny model was trained with 80 classes of objects and in 
our case we'll only have 2 classes (handgun and rifle), so we'll update 
various entries in the configuration file to account for this. Also the minimum 
number of iterations (or batches) is advised to be 2000 per class, but it's 
advised to run many more batches in order to achieve optimal loss/accuracy. 
We'll use 10000 per class in the example below: 
   
    * change the resolution from 416 x 416 to 608 x 608, i.e. `width=608` and 
    `height=608` 
    * change `classes=80` to `classes=2` in each of the two [yolo]-layers
    * change `max_batches` to (classes * 10000), i.e. `max_batches=20000`
    * change `steps` to 80% and 90% of `max_batches`, i.e. `steps=16000,18000`
    * change `filters=255` to `filters=21` in the two [convolutional] layers 
    where this `filters` attribute is present (NOTE: the `filters` value here 
    should equal (\<number of classes\> + 5) * 3), or in the three [convolutional] 
    layers where this attribute is present in the YOLOv3 model configuration

3. Create the file `${DARKNET}/build/darknet/x64/data/obj.names` with 
one object name per line. Use the same order as is used for the object classes in 
the custom dataset. Essentially we can copy the labels file from the Darknet annotated 
dataset, as these should match.

4. Create the file `${DARKNET}/build/darknet/x64/data/obj.data`, containing lines 
specifying the number of object classes (where `classes` = \<number of objects\>), 
location of the training and validation image list files (`train` and `valid`), 
the location of the `obj.names` file we created above (`names`), and the directory 
where weights files will be stored (`backup`). The `train`, `valid`, `names`, and 
`backup` attribute paths are relative to the `${DARKNET}` (darknet repository home) 
directory.
    
    For example:
    ```
    classes=2
    train=build/darknet/x64/data/train.txt
    valid=build/darknet/x64/data/valid.txt
    names=build/darknet/x64/data/obj.names
    backup=backup/
    ```

5. Since the configuration file we're using originally had an input resolution 
(width and height) of 416 x 416 and we've modified our version to 604 x 604 we 
will also need to update the anchors configuration setting. We'll use the `darknet` 
executable to calculate the appropriate anchors for this resolution on our dataset, 
and then copying/pasting the computed anchor values as the `anchors` setting in 
the configuration file `${DARKNET}/cfg/yolov3-tiny-weapons-608.cfg`
    ```bash
    $ ./darknet detector calc_anchors ${DARKNET}/build/darknet/x64/data/obj.data -num_of_clusters 6 -width 608 -height 608 -show
    ```
     
6. Download the pre-trained weights for the convolutional layers and put into the 
directory `${DARKNET}/build/darknet/x64`.
  
    For YOLOv3-tiny:
    ```bash
    $ cd ${DARKNET}/build/darknet/x64
    $ wget https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing
    $ cd ${DARKNET}
    $ ./darknet partial cfg/yolov3-tiny-weapons-608.cfg build/darknet/x64/yolov3-tiny.weights yolov3-tiny.conv.15 15
    ```
    For YOLOv3:
    ```bash
    $ cd ${DARKNET}/build/darknet/x64
    $ wget https://pjreddie.com/media/files/darknet53.conv.74
    ```
   
## Training (transfer learning)
Train the model on a single GPU.

###### YOLOv3-tiny:
```bash
$ cd ${DARKNET}
$ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-tiny-weapons-608.cfg yolov3-tiny.conv.15
```  

###### YOLOv3:
```bash
$ cd ${DARKNET}
$ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-obj.cfg darknet53.conv.74
```  

In order to monitor the training we can add the following command line options 
to the training commands above: `-dont_show -mjpeg_port 8090 -map`
This will allow us to then point a browser to http://localhost:8090/ to monitor the progress.

As the model is training it will save the trained weights at every 1000 
iterations into the `backup` directory specified in the file 
`${DARKNET}/build/darknet/x64/data/obj.data`. For example, after completion of 
2000 batches:
```bash
$ ls -l ${DARKNET}/backup
-rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 22:09 yolov3-tiny-weapons-608_1000.weights
-rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 23:13 yolov3-tiny-weapons-608_2000.weights
-rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 23:13 yolov3-tiny-weapons-608_final.weights
-rw-rw-r--  1 ubuntu ubuntu 34714236 Sep 18 23:13 yolov3-tiny-weapons-608_last.weights
```
_Resume Training_:

If the training is stopped and we want to resume training using the same dataset 
and model configuration then we can restart using the same command used initially 
but with the latest saved weights file as the final argument instead of the pre-trained 
weights file we used in the initial train command. For example, if the YOLOv3 
training is stopped after 2000 iterations and we want to resume then we'd use the 
following training command:
```bash
$ cd ${DARKNET}
$ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-obj.cfg backup/yolov3-obj_2000.weights
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
weights file and specifying the GPU IDs with the `-gpus` option.
    
    YOLOv3-tiny:
    ```bash
    $ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-tiny-weapons-608.cfg backup/yolov3-tiny-weapons-608_2000.weights -gpus 0,1,2,3
    ``` 
    YOLOv3:
    ```bash
    $ ./darknet detector train build/darknet/x64/data/obj.data cfg/yolov3-obj.cfg backup/yolov-obj_2000.weights -gpus 0,1,2,3
    ``` 

## Utilize the trained model for object detection
#### Object detection (inference) on image files
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
