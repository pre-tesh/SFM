6 Photos
   │
   ├─→ Step 0: Read K from EXIF
   │
   ├─→ Step 1: SIFT → keypoints per image
   │
   ├─→ Step 2: Match keypoints between image pairs → (x1, x2)
   │
   ├─→ Step 3: RANSAC 8-point → Fundamental Matrix F
   │
   ├─→ Step 4: F + K → Essential Matrix E
   │
   ├─→ Step 5: SVD of E → 4 camera poses
   │
   ├─→ Step 6: Cheirality check → correct pose (R, t)
   │
   ├─→ Step 7: Triangulation → 3D points X  (from images 1+2)
   │
   ├─→ Step 8: Refine X with least squares
   │
   ├─→ Step 9: PnP for each new image → add camera + new points
   │               (repeat for images 3, 4, 5, 6)
   │
   ├─→ Step 10: Bundle Adjustment → refine everything globally
   │
   └─→ Step 11: Visualize 3D point cloud + camera positions
                     ↓
              🏗️ 3D reconstruction!
