# object_detect_darknet
Object detection utilizing Darknet-based object detection models such as YOLOv3.

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
