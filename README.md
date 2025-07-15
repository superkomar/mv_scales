# Motion Vectors Scale computation

## How to launch

This is how to launch it locally
```
usage: main.py [-h] -alg {keypoints,gradient descent} [-frame_1 FRAME_1]
               [-frame_2 FRAME_2] [-mv MV]

options:
  -h, --help                        show this help message and exit
  -alg {keypoints,gradient descent} Choose one of the following algorithms
  -frame_1 FRAME_1                  File path for the first frame
  -frame_2 FRAME_2                  File path for the second frame
  -mv MV                            File path for img with motion vectors
```


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
