import numpy as np

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    pts_shifted = pts - mean
    avg_dist = np.mean(np.sqrt(pts_shifted[:, 0] ** 2 + pts_shifted[:, 1] ** 2))
    scale = np.sqrt(2) / avg_dist

    """
    The  above normalization is the Hartley normalization, which
    is a common choice for normalizing points before estimating the fundamental matrix.
    The idea is to translate the points so that their centroid is at the origin and
    then scale them so that the average distance from the origin is sqrt(2).
    """

    norm_transform = np.array(
        [
            [    scale,       0     ,   -scale * mean[0]  ],
            [     0   ,      scale  ,   -scale * mean[1]  ],
            [     0   ,       0     ,          1          ],
        ]
    )

    pts_h = np.column_stack([pts, np.ones(len(pts))]) # for making the coordinates homogeneous just append 1 to the end of each point.
    pts_norm = (norm_transform @ pts_h.T).T
    pts_norm = pts_norm[:, :2]#dropping the homogenous coordinate 1.

    return pts_norm, norm_transform
    

def eat_pointalgo(pts1,pts2):#cant write 8 :((((
    #here we calculate the fundamental matrix F from the normalized points
    
    #normalixe the points
    pts1_norm , pts1_norm_transform = normalize_points(pts1)
    pts2_norm , pts2_norm_transform = normalize_points(pts2)

    #make the matrix 
    N = len(pts1_norm)
    A = np.zeros((N,9))
    for i in range(N):
        x,y=pts1_norm[i]
        xp,yp=pts2_norm[i]
        A[i]=[ xp*x ,  xp*y , xp ,yp*x ,  yp*y , yp ,x  ,  y  , 1]

    # Decompose the matrix 
    U, S, Vt = np.linalg.svd(A)
    F= Vt[-1].reshape(3,3)

    #making the matrix rank 2 
    U2,S2,Vt2 = np.linalg.svd(F)
    S2[2]=0
    F= U2 @np.diag(S2)@Vt2

    #de normalize
    F=pts2_norm_transform.T@F@pts1_norm_transform 

    return F 
