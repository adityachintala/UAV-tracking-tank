## SORT: Simple Online and Realtime Tracking

!["DeepSORT Architecture"](https://www.researchgate.net/publication/353256407/figure/fig2/AS:1045653165715457@1626314550831/Architecture-of-Deep-SORT-Simple-online-and-real-time-tracking-with-deep-association.jpg "Architecture of DeepSORT")

- SORT is a simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences. It uses a Kalman filter to predict the state of each object and a Hungarian algorithm to associate detections to tracked objects.

- We can use this algorithm to track the objects in the video and then we can use the id of the object to track the object in the video.

- If there exist multiple tanks in the video then we can track a particular tank by using the id of the tank. It hasn't been implemented yet but it can be implemented in the future with the help of a GUI.

<br/>

## Dronekit
<hr>

- DroneKit is a python API for communicating with the Pixhawk flight controller(PX4) and provides us with a high level API which wraps around the ArduPilot API by leveraging the MAVLink protocol.

- The Flight Controller(PixHawk PX4) we used can communicate with the Onboard Computer(Raspberry Pi or Jetson).

- The Onboard Computer can communicate with the Ground Station using the DroneKit API with the use of Telemetery.

- The Ground Station can be a laptop or a mobile phone with enough processing power to recieve the video stream from the camera and process the video stream to detect the tanks.

- After detecting the tanks and calculating the latitude and longitude of the tanks, the Ground Station can send the latitude and longitude of the tanks to the Onboard Computer using the DroneKit API via Telemetry.

<br/>

## GSD
<hr>

- GSD is the Ground Sample Distance which is the distance between the pixels of the image in the real world.

- It basically gives us an idea as to how much area is covered by a single pixel of that particular camera used and for that particular altitude.

- We already have the data such as altitude, focal length, resolution and sensor size of the camera used.

- We can calculate the GSD using the formula:
    
    ```
    GSD = (Altitude x Focal Length) / (Resolution x Sensor Size)
    ```

- We can use this GSD to calculate the distance between the tanks and the drone which will be used to calculate the resultant latitude and longitude of the tanks.

<br/>

## Dataset
<hr>

- We have created a dataset of about 1000+ images of the tank we were about to use for the project and annotated using Roboflow.

- We have used the **YOLOv7** model to train the dataset.

- As this is a single class detection problem, the AP of the model is north of **90%** which is a good sign.