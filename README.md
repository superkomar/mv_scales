# Motion Vectors Scale computation

## How to launch

Inside the *main.py* script, two different sets of input data are supported: files or folders

This is the help for the general arguments

```
usage: main.py [-h] [-app {gradient,keypoints}]
               [-log_level {info,debug}] [-log_to_file]
               {files,folders} ...

options:
  -h, --help            show this help message and exit

Runtime arguments:
  -app {gradient,keypoints}   the approach to be used
  -log_level {info,debug}     logging level
  -log_to_file                flag to redirect logs to a file

Input data source:
  choose which kind of input data will be used

  {files,folders}       kinds of input
    files               work with files only
    folders             work with files from folders
```


This is the help for the **files** sub-command

```
usage: main.py files [-h] -mv_1 MV_1 -mv_2 MV_2 [-frame_1 FRAME_1] [-frame_2 FRAME_2]

options:
  -h, --help        show this help message and exit
  -mv_1 MV_1        File path for the first img with motion vectors
  -mv_2 MV_2        File path for the second img with motion vectors
  -frame_1 FRAME_1  File path for the first frame
  -frame_2 FRAME_2  File path for the second frame
```


This is the help for the **folders** sub-command
```
usage: main.py folders [-h] -mv MV [-frames FRAMES]

options:
  -h, --help      show this help message and exit
  -mv MV          Folder path for motion vector images
  -frames FRAMES  Folder path for frames images
```

## Motion vectors

Each cell in the file (image) contains a vector pointing pixels between two images\
*Vector direction*: from the current frame to the previous frame

**Example:**
We have a two frames (*frame_1* and *frame_2*) and heir corresponding motion vector files (*mv_1*, *mv_2*).
To reconstruct *frame_1* from *frame_2*, we should take *frame_2* and apply the reversed *mv_2*.

The motion vectors are stored as *'.exr'* files

Movement direction format:
* The X-axis is stored in the first index (ex. *mv[y, x, 0]*)
* The Y-axis is stored in the second index (ex. *mv[y, x, 1]*)

## Algorithms API

Every algorithm can calculate the scales based on two different types of input
1) Two frames and the motion vectors between them
2) Two frames of motion vectors

## Keypoints algorithm

- Detect a set of keypoints in the first frame
- Detect a set of keypoints in the second frame
- Find the closest matching points based on "matcher distance" and non-zero movement between the two images
- Normalize the calculated vectors according to the frame dimensions
- Locate the corresponding points on motion vectors image
- Divide the custom vectors by the vectors from the previous step
- Calculate the mean for each axis (x, y)

### Example
Take some files from the "example" folder
- *"frames\02013.exr"* as the first frame
- *"frames\02014.exr"* as the second frame
- *"motion_vectors\02014.exr"* as thi image containing motion vectors

Result
- scale_x = 0.32345110177993774
- scale_y = -0.8623627424240112

## Gradient descent algorithm

- Use one frame as input and another as target
- Repeat the following steps the required number of times:
  - Calculate the loss between the result of applying motion vectors to the input and the target
  - Update the parameters using gradient descent
- Take the final result and divide it by the initial motion vectors
- Calculate the mean for each axis (x, y)


### Example

Take some files from the "example" folder
- *"frames\toyshop_00000.exr"* as the first frame
- *"frames\toyshop_00001.exr"* as the second frame
- *"motion_vectors\toyshop_00001.exr"* as thi image containing motion vectors

Result (frames)
- Scale X = 1.0049039125442505
- scale_y = 1.0036952495574951

Result (motion vectors)
- Scale X = 0.9999997615814209
- scale_y = 0.9999997615814209
