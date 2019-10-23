# object_detect_darknet
Object detection utilizing Darknet-based object detection models such as YOLOv3.

#### Detection on single (static) image files
Perform object detection on all image files in a directory and display the 
images with labelled bounding boxes:
```bash
$ python3 detect_single_image.py --images_dir /data/datasets/weapons/test \
    --weights /home/james/darknet/20191004/yolov3-tiny-weapons-416_final.weights \
    --config /home/james/darknet/20191004/yolov3-tiny-weapons-416.cfg \
    --labels /home/james/darknet/20191004/labels.txt \
    --confidence 0.6
```
