import argparse
import cv2 as cv
import os

from features.sift_extractor import SIFTExtractor
from features.orb_detector import ORBExtractor
import config
from matching.ransac import ransac_F
from geometry.essential import compute_essentialmatrix, decompose_essentialmatrix

def parse_args():
    parser = argparse.ArgumentParser(description = "Structure from Motion")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--extractor", type=str, default="sift", choices=["sift", "orb"])
    parser.add_argument("--n_features", type=int, default=10000)
    parser.add_argument("--output", type=str, default="output")

    return parser.parse_args()

def load_images(folder):
    """
    Load all images from a folder.
    
    Returns:
        list of (filename, image) tuples, sorted by filename
    """
    images = []
    valid_extensions = (".jpg", ".jpeg", ".png")
    
    filenames = [filename for filename in os.listdir(folder) if filename.lower().endswith(valid_extensions)]
    filenames.sort()
    for filename in filenames:
        path = os.path.join(folder, filename)
        image = cv.imread(path)
        if image is not None:
            images.append((filename, image))
    
    print(f"Loaded {len(images)} images from '{folder}'")
    return images

def get_extractor(args):
    if args.extractor == "sift":
        return SIFTExtractor(n_features=args.n_features)
    elif args.extractor == "orb":
        return ORBExtractor(n_features=args.n_features)
    else:
        raise ValueError(f"use either 'sift' or 'orb': {args.extractor}")

def main():
    args      = parse_args()
    extractor = get_extractor(args)
    images    = load_images(args.images)

    print(f"\nUsing extractor: {args.extractor}")
    print("-" * 40)

    # Extract features for all images
    all_keypoints   = []
    all_descriptors = []

    for i, (filename, image) in enumerate(images):
        kp, des = extractor.extract(image)
        all_keypoints.append(kp)
        all_descriptors.append(des)
        print(f"[{i + 1}/{len(images)}] {filename} : {len(kp)} keypoints")

    # Match the first two images (we'll loop over all pairs later)
    from matching.matcher import FeatureMatcher

    matcher = FeatureMatcher()
    good_matches, pts1, pts2 = matcher.match_features(
        all_keypoints[0], all_descriptors[0],
        all_keypoints[1], all_descriptors[1]
    )

    # Visualize the matches and save to output folder
    os.makedirs(args.output, exist_ok=True)

    # Convert our (i, j) tuples to cv2.DMatch so drawMatches can use them
    cv_matches = [cv.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0)
                  for i, j in good_matches]

    match_img = cv.drawMatches(
        images[0][1], all_keypoints[0],
        images[1][1], all_keypoints[1],
        cv_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    out_path = os.path.join(args.output, "matches_1_2.jpg")
    cv.imwrite(out_path, match_img)
    print(f"\n Match visualization saved to: {out_path}")

    F, inlier_mask = ransac_F(pts1, pts2)
    # Keep only inlier points for the next steps
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    print(f"\nFundamental Matrix F:")
    print(F)

    # --- Module 4: Essential Matrix ---
    E = compute_essentialmatrix(F)
    print(f"\nEssential Matrix E:")
    print(E)

    poses = decompose_essentialmatrix(E)

    # --- Module 5: Pick correct pose via cheirality check ---
    from geometry.essential import check_cheirality
    from geometry.triangulation import triangulate_points
    import numpy as np

    print("\nCheirality check (which pose puts points in front of both cameras?):")

    best_count = 0
    best_R = None
    best_t = None

    for i, (R, t) in enumerate(poses):
        count = check_cheirality(R, t, pts1_inliers, pts2_inliers)
        print(f"  Pose {i+1}: {count}/{len(pts1_inliers)} points in front")
        if count > best_count:
            best_count = count
            best_R = R
            best_t = t

    print(f"\n Best pose: {best_count} points in front of both cameras")
    print(f"R =\n{best_R}")
    print(f"t = {best_t}")

    # --- Triangulate final 3D points with the best pose ---
    K = config.K
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([best_R, best_t.reshape(3, 1)])

    points_3d = triangulate_points(P1, P2, pts1_inliers, pts2_inliers)

    print(f"\n Triangulated {len(points_3d)} 3D points!")
    print(f"   X range: [{points_3d[:,0].min():.2f}, {points_3d[:,0].max():.2f}]")
    print(f"   Y range: [{points_3d[:,1].min():.2f}, {points_3d[:,1].max():.2f}]")
    print(f"   Z range: [{points_3d[:,2].min():.2f}, {points_3d[:,2].max():.2f}]")

        # --- Visualization: 3D scatter plot ---
    import matplotlib.pyplot as plt

    # Filter outliers: keep only points within a reasonable range
    # Use median + threshold to remove wild points
    median = np.median(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - median, axis=1)
    threshold = np.percentile(distances, 90)  # keep 90% closest to median
    mask = distances < threshold
    filtered_points = points_3d[mask]

    print(f"\n Plotting {len(filtered_points)}/{len(points_3d)} points (filtered outliers)")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        filtered_points[:, 0],
        filtered_points[:, 1],
        filtered_points[:, 2],
        c=filtered_points[:, 2],  # color by depth
        cmap='viridis',
        s=2
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SfM 3D Reconstruction (2 views)')
    
    plot_path = os.path.join(args.output, "reconstruction_3d.png")
    plt.savefig(plot_path, dpi=150)
    print(f" 3D plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
