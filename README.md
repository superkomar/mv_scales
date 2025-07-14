# Motion Vectors Scale computation

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
- *frames\02013.exr* as the first frame
- *frames\02014.exr* as the second frame
- *motion_vectors\02014.exr* as thi image containing motion vectors

Result
- scale_x = 0.32345110177993774
- scale_y = -0.8623627424240112

## Gradient descent algorithm
