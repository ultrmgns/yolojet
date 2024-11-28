# Those files auto deploy YOLOv11 model with tracking enabled on nvidia jetson orin/orin nx/orin nano/agx orin.

First, freshly install JetPack 6.1 with SDKManager. Make sure to install everything. 
Then, execute this directly on the Jetson host:

```
sudo apt install libblas3
xhost +local:docker
```

Copy the files from this directory and bring up the container

```
sudo docker compose up
```

## __Tracking uses an additional yaml file that segregates different objects of the same type. It adds a approximately 25-30% performance pentaly in BoT-SORT and 10% penalty in ByteTrack algorhythms__

If you'd like to test these yourself, change the yaml file name in the yolo_gsT.py script on row 109
Note: Tracking requires one additional package from Ultralytics which is automatically installed on first run (lapx>=0.5.2), afteward internet connection is never needed.
