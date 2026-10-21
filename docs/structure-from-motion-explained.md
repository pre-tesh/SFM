# Structure from Motion, from pixels to 3D

Structure from Motion, usually shortened to SfM, answers a simple question:

> If we photograph the same scene from several viewpoints, can we recover both the 3D scene and the camera motion?

Yes. SfM estimates two things together:

1. the position and orientation of each camera
2. a sparse set of 3D points seen across the images

This document explains the geometry behind the pipeline implemented in this repository.

## 1. The pinhole camera model

Start with a 3D point in camera coordinates:

\[
P = (X, Y, Z)
\]

For a pinhole camera with focal length \(f\), the point projects to image coordinates:

\[
x = f\frac{X}{Z}
\]

\[
y = f\frac{Y}{Z}
\]

The equations come directly from similar triangles.

For the horizontal direction:

\[
\frac{x}{f} = \frac{X}{Z}
\]

so:

\[
x = f\frac{X}{Z}
\]

The same argument gives the vertical equation.

The division by \(Z\) is what creates perspective. A distant object appears smaller because increasing depth reduces the projected coordinates.

![Pinhole camera geometry](https://commons.wikimedia.org/wiki/Special:FilePath/Pinhole%20Camera%20Diagram.png)

*Pinhole camera geometry. Image by Finitehoax, CC0, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pinhole_Camera_Diagram.png).*

### Why one image cannot give depth

Suppose:

\[
P_1 = (1, 1, 2)
\]

and:

\[
P_2 = (2, 2, 4)
\]

Both have the same ratios \(X/Z\) and \(Y/Z\), so both project to the same pixel.

That means one pixel does not identify one unique 3D point. It identifies a ray extending from the camera into the scene.

This is the first important idea in SfM.

## 2. Camera intrinsics and extrinsics

Real images are measured in pixels, so we introduce the camera intrinsic matrix:

\[
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\]

Here:

- \(f_x\) and \(f_y\) are focal lengths in pixel units
- \(c_x\) and \(c_y\) are the principal point coordinates

A camera can also rotate and translate relative to the world. If \(R\) is the camera rotation and \(t\) is the translation, then a world point \(P_w\) becomes:

\[
P_c = RP_w + t
\]

The complete projection equation is:

\[
s p = K[R|t]P_w
\]

This equation is central to SfM. We want to estimate both the camera parameters and the unknown 3D structure.

## 3. Feature detection

The computer first needs points that can be recognized again in another image.

This repository uses SIFT.

SIFT finds distinctive locations such as:

- corners
- textured patches
- strong local intensity structures

Each keypoint receives a descriptor. A descriptor is a numerical representation of the local visual pattern around that keypoint.

The descriptor allows us to ask:

> Does this feature in image A look like the same physical feature in image B?

## 4. Feature matching

Descriptors from two images are compared to find candidate correspondences.

This repository uses FLANN and Lowe's ratio test.

The ratio test compares the best descriptor match with the second best match. If both are almost equally good, the match is ambiguous and is rejected.

This removes many bad matches, but not all of them.

Repeated windows, tiles, leaves, reflections, and similar textures can still create incorrect correspondences.

That is why geometric verification is needed.

## 5. Epipolar geometry

Consider two cameras observing the same 3D point.

The two camera centers and the 3D point define an epipolar plane. That plane intersects each image plane along an epipolar line.

![Epipolar geometry](https://commons.wikimedia.org/wiki/Special:FilePath/Epipolar%20Geometry1.svg)

*Epipolar geometry. Public domain illustration by ZooFari via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Epipolar_Geometry1.svg).*

The useful result is this:

If we know a point in image 1, its matching point in image 2 cannot be anywhere. It must lie on the corresponding epipolar line.

For homogeneous image points \(x_1\) and \(x_2\):

\[
x_2^T F x_1 = 0
\]

where \(F\) is the fundamental matrix.

If the camera intrinsics are known, we can use the essential matrix \(E\):

\[
x_2^T E x_1 = 0
\]

For calibrated cameras:

\[
E = [t]_\times R
\]

where \(R\) is the relative rotation and \(t\) is the relative translation direction.

## 6. RANSAC rejects outliers

Feature matching always produces some incorrect matches.

RANSAC protects the geometry estimation from those outliers.

A simplified RANSAC loop is:

```text
repeat many times:
    choose a small random subset of matches
    estimate the geometric model
    test all matches against that model
    count the inliers

keep the model with the strongest consensus
```

In this repository, RANSAC is used when estimating the fundamental matrix and during camera registration.

The important idea is that a correct geometric model should explain many matches at once.

## 7. Recover relative camera motion

Once the essential matrix is known, it can be decomposed to obtain candidate relative camera rotations and translations.

There is one important limitation.

Monocular SfM cannot recover absolute scale from images alone.

For example, if the real camera moved 10 cm, a reconstruction scaled by 10, 100, or 0.1 could still produce the same image projections as long as every camera and 3D point is scaled consistently.

So the translation direction is recoverable, but its physical magnitude is not known without extra information.

Metric scale can come from sources such as:

- a known object size
- a stereo baseline
- GPS
- an IMU
- depth measurements

## 8. Triangulation

Once two camera poses are known and a feature correspondence is available, we can reconstruct the 3D point.

Each observation defines a ray from a camera center into the scene.

In ideal geometry, the two rays intersect at one point.

![Ideal triangulation](https://commons.wikimedia.org/wiki/Special:FilePath/TriangulationIdeal.svg)

*Ideal two view triangulation. Public domain illustration via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:TriangulationIdeal.svg).*

For a camera matrix \(P_1\) and image point \(x_1\):

\[
x_1 \times (P_1X) = 0
\]

The second camera gives another constraint:

\[
x_2 \times (P_2X) = 0
\]

These equations can be rearranged into a linear system:

\[
AX = 0
\]

The homogeneous 3D point \(X\) can be estimated with singular value decomposition.

In real data, the rays usually do not intersect perfectly because image measurements and camera poses contain noise. Triangulation therefore finds the point that best satisfies the observations.

## 9. Reprojection error

After reconstructing a 3D point, we can project it back into the image.

If \(X_j\) is a 3D point and camera \(i\) predicts image position:

\[
\hat{x}_{ij} = \operatorname{project}(P_i, X_j)
\]

while the observed feature is \(x_{ij}\), then the reprojection error is:

\[
e_{ij} = \lVert x_{ij} - \hat{x}_{ij} \rVert
\]

A small reprojection error means the reconstructed geometry is consistent with the measured image feature.

A large error can indicate:

- a bad feature match
- an inaccurate camera pose
- a poor 3D point estimate
- incorrect calibration

## 10. Bundle adjustment

Early estimates are noisy. Bundle adjustment jointly improves the cameras and 3D points.

A simplified objective is:

\[
\min_{P_i, X_j}
\sum_{i,j}
\left\|
x_{ij} - \operatorname{project}(P_i, X_j)
\right\|^2
\]

This is a nonlinear least squares problem.

Instead of refining each point independently, bundle adjustment optimizes the whole reconstruction so that all camera observations agree as well as possible.

This is implemented in `bundle_adjustment.py`.

## 11. Adding more cameras with PnP

After the first image pair creates an initial 3D map, more images can be registered.

At this stage some 3D points are already known, and the new image contains 2D observations of them.

The problem becomes:

```text
known:
    3D point X1 -> image point x1
    3D point X2 -> image point x2
    3D point X3 -> image point x3
    ...

unknown:
    camera rotation R
    camera translation t
```

This is the Perspective n Point problem, usually called PnP.

PnP estimates the new camera pose. RANSAC is commonly used around PnP to reject bad 2D to 3D correspondences.

After the camera is registered, additional points can be triangulated between that camera and previously registered views.

## 12. The complete incremental SfM loop

The complete pipeline can be viewed as:

```text
input images
    |
    v
SIFT keypoints and descriptors
    |
    v
feature matching
    |
    v
geometric verification with RANSAC
    |
    v
choose a seed image pair
    |
    v
estimate relative pose
    |
    v
triangulate initial points
    |
    v
bundle adjustment
    |
    v
choose another image
    |
    v
register camera with PnP
    |
    v
triangulate additional points
    |
    v
bundle adjustment again
    |
    v
repeat
    |
    v
camera poses + sparse 3D point cloud
```

The reconstruction grows incrementally because every solved camera creates new opportunities to triangulate points, and every new 3D point helps register more cameras.

## 13. Why camera movement matters

SfM depends on parallax.

If the camera only rotates around its optical center, image features move, but there is very little information about their depth.

Translation creates viewpoint differences that allow triangulation.

There is a tradeoff:

- too little movement gives weak depth estimates
- too much movement makes feature matching difficult
- overlapping views with moderate viewpoint changes usually work best

## 14. Why SfM can fail

SfM is powerful, but it relies on stable visual geometry.

Common difficult cases include:

- textureless walls
- repeated patterns
- moving objects
- reflective surfaces
- transparent objects
- motion blur
- poor camera calibration
- tiny camera baselines
- extremely large viewpoint changes

## 15. SfM is sparse reconstruction

SfM usually reconstructs only distinctive feature points.

The result is therefore a sparse point cloud, not a dense surface.

A typical full 3D reconstruction stack is:

```text
images
  -> Structure from Motion
       -> camera poses + sparse point cloud
  -> Multi View Stereo
       -> dense point cloud
  -> meshing
       -> surface
  -> texturing
       -> textured 3D model
```

This repository implements the sparse SfM part of that stack.

## 16. The mental model to remember

The easiest way to understand SfM is as a feedback loop between appearance and geometry:

1. image features suggest correspondences
2. geometry rejects inconsistent matches
3. correspondences reveal camera motion
4. camera motion enables triangulation
5. triangulated points help register more cameras
6. bundle adjustment improves everything together

A single image says:

> The point lies somewhere along this viewing ray.

A second image gives another ray.

Once the relative cameras are known, triangulation estimates where those rays meet in 3D.

Repeat this across many images and many features, and ordinary photographs become a 3D reconstruction.

## Relation to this repository

The main implementation files correspond closely to the theory above:

| Concept | Implementation |
| --- | --- |
| SIFT feature extraction | `sift_extractor.py` |
| Feature matching | `matcher.py` |
| RANSAC | `ransac.py` |
| Multi view geometry | `geometry/` |
| Incremental reconstruction | `main.py` |
| Bundle adjustment | `bundle_adjustment.py` |
| Camera calibration | `config.py` |

For the shorter implementation overview and setup instructions, return to the [main README](../README.md).
