# Motion Vectors Scale computation

## How to launch

This is how to launch it locally
```
usage: main.py [-h] [-app {keypoints,gradient,all}]
               [-frame_1 FRAME_1] [-frame_2 FRAME_2]
               [-mv_1 MV_1] [-mv_2 MV_2]
               [-log_level {DEBUG,INFO}]

options:
  -h, --help                        show this help message and exit
  -alg {keypoints,gradient}         Choose one of the following algorithms
  -frame_1 FRAME_1                  File path for the first frame
  -frame_2 FRAME_2                  File path for the second frame
  -mv_1 MV                          File path for the first img with motion vectors
  -mv_2 MV                          File path for the second img with motion vectors
  -log_level {DEBUG,INFO}           Logging level
```

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
