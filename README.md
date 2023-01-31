# Military Tank object Detection and Tracking using UAV

<p align="center">
    <img src="https://github.com/adityachintala/UAV-tracking-tank/blob/main/img/drone_flying.gif?raw=true" alter="Drone flying">
</p>

## Problem Statement

An autonomous navigation by a UAV towards it's target such as another military tank or vehicles - based on the detection, locking-on and following the movements of the target is a much needed application. A UAV flying with a companion computer and HD camera shall detect military tanks and vehicles on the ground and follow them.


## Solution

<p align="center">
  <img src="https://github.com/adityachintala/UAV-tracking-tank/blob/main/img/drone_2.jpg?raw=true" alter="Steps">
</p>

- The idea is to receive the video stream from the camera(radio waves), run detections on it, perform calculations and then send the coordinates of the target to the flight controller(using telemetry). 

- We will be using a PixHawk 4 flight controller and a companion computer (Raspberry Pi 4) to control the drone

- This RPi will be connected to the drone via a USB cable and the drone will be equipped with a HD camera(Raspberry Pi Camera Module V2)

- The camera will be connected to the companion computer and the video will be streamed to the Ground Station.

- Mission Planner is used to check errors, resolve them, and check GPS satellites before flight.

- The Ground Station will be running a python script to detect and track the target, which uses YOLOv7 to detect the target and SORT to track the target.

- The Ground Station will be sending the coordinates of the target to the companion computer, which will be sending the coordinates to the Pixhawk flight controller.

- The PixHawk PX4 flight controller will be sending the coordinates to the drone.

- The drone will be flying towards the target and will be following it.

## Workflow

<p align="center">
    <img src="https://github.com/adityachintala/UAV-tracking-tank/blob/main/img/workfloww.jpg?raw=true" alter="Workflow">
</p>

## YOLOv7

<p align="center">
    <img src="https://github.com/adityachintala/UAV-tracking-tank/blob/main/img/yolov7.jpg?raw=true" alter="yolov7">
</p>

- The aim behind the implementation of YOLOv7 is to achieve better accuracy as compared with YOLOR, YOLOv5, and YOLOX.

- YOLOv7 is a real-time object detection model that detects 80 different classes. It is a state-of-the-art object detection model that is fast and accurate. The 
development of YOLOv7 is completely in PyTorch.

- It is a convolutional neural network that is 49 layers deep. It can run in real-time at 30 FPS.

- It is a one-stage detector that directly predicts bounding boxes and class probabilities for those boxes. It is a very small model(only 17 MB) and very fast.

- It is also very accurate, achieving 44.8 mAP on COCO test-dev.

## SORT: Simple Online and Realtime Tracking

<p>
    <img src="https://github.com/adityachintala/UAV-tracking-tank/blob/main/img/SORT.png?raw=true" alter="SORT">
</p>

- SORT is a simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences. It uses a Kalman filter to predict the state of each object and a Hungarian algorithm to associate detections to tracked objects.

- We can use this algorithm to track the objects in the video and then we can use the id of the object to track the object in the video.

- If there exist multiple tanks in the video then we can track a particular tank by using the id of the tank. It hasn't been implemented yet but it can be implemented in the future with the help of a GUI.

## Dronekit

- DroneKit is a python API for communicating with the Pixhawk flight controller(PX4) and provides us with a high level API which wraps around the ArduPilot API by leveraging the MAVLink protocol.

- The Flight Controller(PixHawk PX4) we used can communicate with the Onboard Computer(Raspberry Pi or Jetson).

- The Onboard Computer can communicate with the Ground Station using the DroneKit API with the use of Telemetry.

- The Ground Station can be a laptop or a mobile phone with enough processing power to receive the video stream from the camera and process the video stream to detect the tanks.

- After detecting the tanks and calculating the latitude and longitude of the tanks, the Ground Station can send the latitude and longitude of the tanks to the Onboard Computer using the DroneKit API via Telemetry.

## GSD

<p align = "center"r>
    <img src="https://github.com/adityachintala/UAV-tracking-tank/blob/main/img/gsd.jpg?raw=true" alt="GSD">
</p>

- GSD is the Ground Sample Distance which is the distance between the pixels of the image.

- It basically gives us an idea as to how much area is covered by a single pixel of that particular camera used and for that particular altitude.

- We already have the data such as altitude, focal length, resolution and sensor size of the camera used.

- We can calculate the GSD using the formula:
    ```
    GSD = (Altitude x Focal Length) / (Resolution x Sensor Size)
    ```

- We can use this GSD to calculate the distance between the tanks and the drone which will be used to calculate the resultant latitude and longitude of the tanks.

## Dataset

- We have created a dataset of about 1000+ images of the tank we were about to use for the project and annotated using Roboflow.

- We have used the **YOLOv7** model to train the dataset.

- As this is a single class detection problem, the AP of the model is north of **90%** which is a good sign.

- The model is able to detect the tank with a good accuracy.

## Geopy

- Geopy is a python library which is used to calculate the distance between two points on the earth.

- We can use this library to calculate the distance between the drone and the tank.

- This can also be used to calculate the latitude and longitude of the tank, given the distance between the drone and the tank.

## Yolo + SORT + Geopy

- We have used the YOLOv7 model to detect the tanks, SORT algorithm to track the tanks and Geopy to calculate the latitude and longitude of the tanks.

- Once we locate the tanks, we can send the latitude and longitude of the tanks to the drone using the DroneKit API.

- The tanks are tracked in real time and the drone is able to follow the tanks.

- The drone is able to return to the launch point in case of any failures.


## Implementation

- clone the repository

    ```
     git clone https://github.com/adityachintala/UAV-tracking-tank
    ```

- install the requirements

    ```
    pip install -r requirements.txt
    ```

- run the script
    ``` bash
    python detect_and_track.py --source 0 --weights best.pt --no-download --baud 57600 --altitude 4 --connect com3 --view-img
    ```

    - (source can be a video file or a webcam, use 0/1/2 as the source for webcam)
    - (weights can be the path to the weights file)
    - (baud is the baud rate of the pixhawk flight controller)
    - (altitude is the altitude at which the drone will fly)
    - (connect is the port to which the pixhawk flight controller is connected)
    - (view-img is used to view the output video)

## References and Articles

- [Yolov7 Training on custom data](https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623)

- [YOLOv7 object tracking](https://github.com/RizwanMunawar/yolov7-object-tracking)

- [SORT: Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)

- [Object tracking using DeepSORT](https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/)

- [DroneKit](https://dronekit.io/)

- [Geopy](https://medium.com/featurepreneur/geopy-dfc65e40f4c9)

## Applications

- The application of this project is to detect and track military tanks and vehicles on the ground and follow them.

- This can be used in military operations to detect and track enemy tanks and follow them.

- This can also be used in wildlife conservation to detect and track animals.

## Conclusion

We have been able to:

- Detect and track the tanks in real time

- Send the latitude and longitude of the tanks to the drone

- Follow the tanks in real time

- Return to launch point, incase of any failures
