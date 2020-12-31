# Deepsort yolov5 in ros melodic1

## Environment 
- Ubuntu 18.04, cuda 10.2, RTX 2070 super, Ros melodic 1

Associative Repository : How to use ROS

## Direction
If you created an ROS workspace(ex. catkin_ws), you put the ```git clone https://github.com/winston1214/Deepsort_yolov5_ros.git``` in workspace
```
$ cd catkin_ws/src
$ git clone https://github.com/winston1214/Deepsort_yolov5_ros.git
$ cd ..
$ catkin_make
```

## Train

<a href='https://github.com/winston1214/AICT/tree/master/yolov5'>YOLOv5 Training</a>

- Simple weight file(small_best.pt)
  
  : pretraining weight model - yolov5s.pt(As the size of the weight model increases, real-time performance decreases.)
  
  : batch size - 8, epochs - 150 ,classes - person, car

## Detect

- ROS Image Topic Subscriber

```
$ roscore
$ cd catkin_ws/src/Deepsort_yolov5_ros
$ python3 ros_track.py --source ros --weights weights/small_best.pt
```

- Image, Video, webcam etc..

```
$ python3 ros_track.py --source file_path --weights weights/small_best.pt
```

