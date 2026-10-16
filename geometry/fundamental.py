import numpy as np

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    pts_shifted = pts - mean
    avg_dist = np.mean(np.sqrt(pts_shifted[:, 0] ** 2 + pts_shifted[:, 1] ** 2))
    if not np.isfinite(avg_dist) or avg_dist < 1e-12:
        raise ValueError("Cannot normalize coincident/nonfinite points")
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
    if len(pts1) < 8 or np.shape(pts1) != np.shape(pts2):
        raise ValueError("Eight-point estimation needs at least 8 paired points")
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
        A[i]=[ xp*x ,  xp*y , xp ,yp*x ,  yp*y , yp ,x  ,  y  , 1] # this a is the a.f =0 wali matrix , so we stack up 8 instances of it and find from f11 to f33.

    """ 
        WE TOTALLLY FUCKING EXPLOIT CLEVER MATHS AND APPROXIMATE F.
        what we do below is a clever trick :
        1. we ned to find f such that a.f=0;
        2. so we svd (A) now we get 3 things u,s,vt; 
        3. vt would give us 9 orhtogonal vectors , 
        from the direction A points in the most to 
        the directiion A points the least .
        watch visual kernel's svd video , i love the visuals .
        4. now , as we want A.F to be 0 (or the least considering some error in sift points )
        A·f is small when f is the direction the rows point along the least , 
        so we take the last row of vt and reshape it into a 3x3 matrix and we can use it as F :)
        ABSOLUTERLY FUCKING GENIUS , I LOVE THIS TRICK.

        Now the second , trick that we exploit is that we know F is rank 2 ,( just assume , dont think about why is it rank 2 ) 
        which means det(F) = 0 ;
        A matrix is rank 2 exactly when it has only two layers.
        so we svd(F) and set the last collumn of S to 0,0,0 .
        and again to build F we do  reverse svd and get F .

        next we de normalize the points by multiplying the normalization matrices to F and we get the final F.

    """


    # Decompose the matrix 
    # 8x9 needs the full right nullspace; larger fits do not need a huge NxN U.
    U, S, Vt = np.linalg.svd(A, full_matrices=(N < 9))
    F= Vt[-1].reshape(3,3)

    #making the matrix rank 2 
    U2,S2,Vt2 = np.linalg.svd(F)
    S2[2]=0
    F= U2 @np.diag(S2)@Vt2

    #de normalize
    F=pts2_norm_transform.T@F@pts1_norm_transform 

    return F 
