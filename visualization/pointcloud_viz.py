import numpy as np


def export_ply(points_3d, filepath, colors=None):
    """
    Export 3D points to a .ply file that can be opened in MeshLab.
    
    PLY is a simple text format:
        header → then one line per point with x, y, z, r, g, b
    
    Args:
        points_3d: (N, 3) array
        filepath: output .ply file path
        colors: optional (N, 3) array of RGB values (0-255)
    """
    N = len(points_3d)
    
    if colors is None:
        # Color by depth (Z value) — map to a blue→red gradient
        z = points_3d[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)  # normalize to 0-1
        colors = np.zeros((N, 3), dtype=np.uint8)
        colors[:, 0] = (z_norm * 255).astype(np.uint8)         # red channel
        colors[:, 1] = ((1 - np.abs(z_norm - 0.5) * 2) * 255).astype(np.uint8)  # green
        colors[:, 2] = ((1 - z_norm) * 255).astype(np.uint8)   # blue channel
    
    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # One line per point
        for i in range(N):
            x, y, z = points_3d[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"[PLY] Exported {N} points to: {filepath}")