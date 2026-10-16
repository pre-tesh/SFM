import numpy as np
from geometry.fundamental import eat_pointalgo


def sampson_distance_squared(F, pts1_h, pts2_h):
    """Scale-invariant first-order geometric error in squared pixel units."""
    lines2 = pts1_h @ F.T
    lines1 = pts2_h @ F
    numerator = np.sum(pts2_h * lines2, axis=1) ** 2
    denominator = np.sum(lines2[:, :2] ** 2 + lines1[:, :2] ** 2, axis=1)
    return np.divide(numerator, denominator, out=np.full(len(pts1_h), np.inf),
                     where=denominator > np.finfo(float).tiny)


def ransac_F(pts1, pts2, threshold=2.0, max_iterations=1000, seed=42):
    """Estimate F by repeatedly fitting 8 random matches and keeping most inliers."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2 or len(pts1) < 8:
        raise ValueError('RANSAC needs at least 8 paired 2D points')

    n = len(pts1)
    pts1_h = np.column_stack((pts1, np.ones(n)))
    pts2_h = np.column_stack((pts2, np.ones(n)))
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    
    for _ in range(max_iterations):
        indices = rng.choice(n, 8, replace=False)
        try:
            F = eat_pointalgo(pts1[indices], pts2[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue

        errors = sampson_distance_squared(F, pts1_h, pts2_h)
        mask = errors < threshold ** 2
        if mask.sum() > best_mask.sum():
            best_mask = mask

    if best_mask.sum() < 8:
        raise ValueError('RANSAC could not find 8 geometrically consistent matches')

    F = eat_pointalgo(pts1[best_mask], pts2[best_mask])
    mask = sampson_distance_squared(F, pts1_h, pts2_h) < threshold ** 2
    return F, mask
