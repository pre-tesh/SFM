import numpy as np
#CAMERA PARAMETERS (DEPENDENT ON THE CAMERA USED )

#intrinsic matrix of the camera
K = np.array([
    # Row 1: [f, 0, cx]
    # Row 2: [0, f, cy]
    # Row 3: [0, 0,  1]
    #
    # Image resolution: 3456 x 4608 (WxH)
    # Focal length estimated from ~70 deg HFOV on a 3456px-wide sensor:
    #   f = W / (2 * tan(HFOV/2)) ≈ 2468 px
    # Principal point at image centre: cx = W/2 = 1728, cy = H/2 = 2304
    [2468,    0, 1728],
    [   0, 2468, 2304],
    [   0,    0,    1]
], dtype=np.float64)

MIN_MATCHES = 8
