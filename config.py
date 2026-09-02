import numpy as np
#CAMERA PARAMETERS (DEPENDENT ON THE CAMERA USED )

#intrinsic matrix of the camera
K = np.array([
    # Row 1: [f, 0, cx]
    # Row 2: [0, f, cy]  
    # Row 3: [0, 0,  1]
    [350,   0, 480],
    [  0, 350, 270],
    [  0,   0,   1]
], dtype=np.float64)

#PIPELINE PARAMETERS (INDEPENDENT OF THE CAMERA USED )

RATIO_THRESHOLD = 0.75
RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 0.1
MIN_MATCHES = 8
