"""
Interactive 3D viewer for SfM reconstruction output.

Reads the .ply point cloud and optional cameras.npz, then generates
a self-contained HTML file you can open (and rotate/zoom/pan) in any browser.

Standalone usage:
    python visualization/viewer.py --ply output_bulbasaur2/reconstruction.ply \
                                   --cameras output_bulbasaur2/cameras.npz

The HTML is also generated automatically at the end of every main.py run.
"""

import numpy as np
import argparse
import os
import webbrowser

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# PLY reader (ASCII and binary vertex data; no extra deps needed)
# ---------------------------------------------------------------------------

def read_ply(filepath):
    """Read ASCII or binary PLY vertices, including COLMAP's binary export.

    Returns (N, 3) float64 coordinates and (N, 3) uint8 RGB colors.
    Scalar vertex properties are read by name; trailing faces are ignored.
    """
    type_map = {
        'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2',
        'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8',
        'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2',
        'int32': 'i4', 'uint32': 'u4', 'float32': 'f4', 'float64': 'f8',
    }
    with open(filepath, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Not a PLY file')
        fmt = None
        count = None
        element = None
        properties = []
        while True:
            raw_line = f.readline()
            if not raw_line:
                raise ValueError('Truncated PLY header: missing end_header')
            fields = raw_line.decode('ascii').split()
            if not fields or fields[0] in ('comment', 'obj_info'):
                continue
            if fields[0] == 'format':
                if len(fields) != 3 or fields[2] != '1.0':
                    raise ValueError('Unsupported PLY format/version')
                fmt = fields[1]
            elif fields[0] == 'element':
                element = fields[1]
                element_count = int(fields[2])
                if element_count < 0:
                    raise ValueError('Negative PLY element count')
                if element == 'vertex':
                    if count is not None:
                        raise ValueError('Duplicate PLY vertex element')
                    count = element_count
                elif count is None and element_count > 0:
                    raise ValueError('PLY vertex data must precede other nonempty elements')
            elif fields[0] == 'property' and element == 'vertex':
                if len(fields) != 3 or fields[1] not in type_map:
                    raise ValueError('Only supported scalar PLY vertex properties can be read')
                properties.append((fields[2], type_map[fields[1]]))
            elif fields[0] == 'end_header':
                break

        formats = {'ascii': '=', 'binary_little_endian': '<', 'binary_big_endian': '>'}
        if fmt not in formats or count is None:
            raise ValueError('PLY must declare a supported format and vertex element')
        names = [name for name, _ in properties]
        if len(set(names)) != len(names) or not all(name in names for name in ('x', 'y', 'z')):
            raise ValueError('PLY vertices need unique properties including x, y, z')
        dtype = np.dtype([(name, formats[fmt] + kind) for name, kind in properties])
        if count == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
        if fmt == 'ascii':
            vertices = np.loadtxt(f, dtype=dtype, max_rows=count, ndmin=1)
        else:
            vertices = np.fromfile(f, dtype=dtype, count=count)
        if len(vertices) != count:
            raise ValueError(f'Truncated PLY vertex data: expected {count}, got {len(vertices)}')
        points = np.column_stack([vertices[name] for name in ('x', 'y', 'z')]).astype(np.float64)
        if all(name in names for name in ('red', 'green', 'blue')):
            colors = np.column_stack([vertices[name] for name in ('red', 'green', 'blue')]).astype(np.uint8)
        else:
            colors = np.full((count, 3), 180, dtype=np.uint8)
        return points, colors


# ---------------------------------------------------------------------------
# Main visualize function
# ---------------------------------------------------------------------------

def visualize(ply_path, cameras_path=None, output_html=None,
              open_browser=True, max_points=60000):
    """
    Build an interactive Plotly 3D scene from a PLY file + optional cameras.

    Args:
        ply_path     : path to reconstruction.ply
        cameras_path : path to cameras.npz  (optional; shows camera centres)
        output_html  : where to write the HTML file (default: next to PLY)
        open_browser : auto-open in default browser after writing
        max_points   : subsample threshold (Plotly slows above ~100k pts)
    """
    if not PLOTLY_AVAILABLE:
        print("[Viewer] plotly is not installed. Run:  pip install plotly")
        return None

    print(f"[Viewer] Reading {ply_path} ...")
    points, colors = read_ply(ply_path)
    n_total = len(points)
    if n_total == 0:
        raise ValueError('PLY contains no vertices to visualize')
    if max_points < 1:
        raise ValueError('max_points must be positive')
    print(f"[Viewer] Loaded {n_total:,} points")

    # Subsample if needed so the browser stays responsive
    if n_total > max_points:
        idx = np.random.default_rng(42).choice(n_total, max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        print(f"[Viewer] Subsampled to {max_points:,} points for interactive display")

    # Plotly expects colour as 'rgb(r,g,b)' strings
    color_strs = [f'rgb({r},{g},{b})' for r, g, b in colors]

    fig = go.Figure()

    # ── Point cloud ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=color_strs,
            opacity=0.85,
        ),
        name=f'Point cloud  ({len(points):,} pts)',
        hovertemplate=(
            'x: %{x:.3f}<br>'
            'y: %{y:.3f}<br>'
            'z: %{z:.3f}<extra></extra>'
        ),
    ))

    # ── Camera centres ────────────────────────────────────────────────────────
    if cameras_path and os.path.exists(cameras_path):
        data = np.load(cameras_path, allow_pickle=True)
        centers = data['centers']          # (N_cams, 3)
        cam_labels = [f'Cam {i+1}' for i in range(len(centers))]

        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers+text',
            marker=dict(size=6, color='red', symbol='diamond', opacity=1.0),
            text=cam_labels,
            textposition='top center',
            textfont=dict(size=8, color='red'),
            name=f'Cameras  ({len(centers)})',
            hovertemplate=(
                '%{text}<br>'
                'x: %{x:.3f}<br>'
                'y: %{y:.3f}<br>'
                'z: %{z:.3f}<extra></extra>'
            ),
        ))
    else:
        if cameras_path:
            print(f"[Viewer] cameras.npz not found at {cameras_path} — skipping camera overlay")

    # ── Layout ────────────────────────────────────────────────────────────────
    cam_count_str = ''
    if cameras_path and os.path.exists(cameras_path):
        try:
            cam_count_str = f'  |  {len(np.load(cameras_path, allow_pickle=True)["centers"])} cameras'
        except Exception:
            pass

    fig.update_layout(
        title=dict(
            text=f'SfM Reconstruction — {len(points):,} / {n_total:,} points{cam_count_str}',
            font=dict(size=17, color='white'),
            x=0.5,
        ),
        scene=dict(
            aspectmode='data',
            bgcolor='rgb(8, 8, 18)',
            xaxis=dict(
                title=dict(text='X', font=dict(color='lightgray')),
                backgroundcolor='rgb(8,8,18)',
                gridcolor='rgba(100,100,120,0.4)',
                showbackground=True,
                tickfont=dict(color='lightgray'),
            ),
            yaxis=dict(
                title=dict(text='Y', font=dict(color='lightgray')),
                backgroundcolor='rgb(8,8,18)',
                gridcolor='rgba(100,100,120,0.4)',
                showbackground=True,
                tickfont=dict(color='lightgray'),
            ),
            zaxis=dict(
                title=dict(text='Z', font=dict(color='lightgray')),
                backgroundcolor='rgb(8,8,18)',
                gridcolor='rgba(100,100,120,0.4)',
                showbackground=True,
                tickfont=dict(color='lightgray'),
            ),
        ),
        paper_bgcolor='rgb(12, 12, 22)',
        plot_bgcolor='rgb(12, 12, 22)',
        font=dict(color='white'),
        legend=dict(
            font=dict(color='white', size=11),
            bgcolor='rgba(255,255,255,0.05)',
            bordercolor='rgba(255,255,255,0.15)',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # ── Write HTML ────────────────────────────────────────────────────────────
    if output_html is None:
        output_html = ply_path.replace('.ply', '_viewer.html')

    # include_plotlyjs='cdn' keeps the file small (~10 KB vs ~3 MB)
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"[Viewer] Interactive viewer saved → {output_html}")

    if open_browser:
        webbrowser.open(f'file://{os.path.abspath(output_html)}')
        print(f"[Viewer] Opened in browser")

    return output_html


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for SfM PLY output"
    )
    parser.add_argument("--ply", type=str, required=True,
                        help="Path to reconstruction.ply")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Path to cameras.npz (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: <ply_name>_viewer.html)")
    parser.add_argument("--max-points", type=int, default=60000,
                        help="Max points rendered interactively (default: 60000)")
    parser.add_argument("--no-open", action="store_true",
                        help="Write HTML but don't open the browser")
    args = parser.parse_args()

    visualize(
        ply_path=args.ply,
        cameras_path=args.cameras,
        output_html=args.output,
        open_browser=not args.no_open,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
