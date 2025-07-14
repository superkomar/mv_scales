# Motion Vectors Scale computation

## Keypoints algorithm

- Detect a set of keypoints in the first image
- Detect a set of keypoints in the second image
- Find the closest matching points based on "matcher distance" and non-zero movement between the two images
- Normalize the calculated vectors according to the image dimensions
- Locate the corresponding points on motion vectors image
- Divide the custom vectors by the vectors from the previous step
- Calculate the mean for each axis (x, y)

## Gradient descent algorithm
