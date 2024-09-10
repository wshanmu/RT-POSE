# 4Dradar_Processing
## Step 1
Using Matlab to prepocss raw mmWave data to 4D radar tensor with complex format in `Doppler(64) x Range(256) x Azimuth(128)x Elevation(32)`. In this work, the program is running under **Matlab2022b**.
* Open `main.m` in folder `mmWave-Matlabe`
* Change the `Your Data Path` in `line 38` in the main.m
* **Optional**: change the `line 43` to specific sequences you want to process
* Run `main.m`
* Processed data will be stored in `(Your Data Path)/RT-Pose/sequences/ (sequnece ID) /radar/mat`

## Step 2
Using Pyhton to prepocss `.mat file` to **cartesian coordinate** in `Doppler(64) x Z(32) x Y(128) x X(256)` and store as `.npy file`. 

```
conda create -n 4Dradar_preprocess python=3.9
conda activate 4Dradar_preprocess
pip install -r requirements.txt
```
Process all sequences:
```
python 4Dradar2xyz.py --dataset_dir (Your Data Path) 
```
**Optional**: Process specific sequences you want:
```
python 4Dradar2xyz.py --dataset_dir (Your Data Path) --sequence (sequnece ID)
```
