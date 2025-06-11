# simple_grabcut.py
"""
A minimal, self‑contained re‑implementation of the GrabCut algorithm that avoids
OpenCV's built‑in `cv2.grabCut`.  It relies only on

* NumPy – array maths
* scikit‑learn – GaussianMixture for colour modelling
* PyMaxflow – graph‑cut / max‑flow optimisation
* scikit‑image – I/O convenience (can be replaced by imageio/Pillow)

Quick start  
$ pip install numpy scikit‑learn PyMaxflow scikit‑image  
$ python simple_grabcut.py input.jpg output.png --rect 50 50 200 300 --iter 5

The script will write an RGBA PNG whose alpha channel comes from the estimated
foreground mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
import maxflow
from skimage import io

# -----------------------------------------------------------------------------
#  Core helpers
# -----------------------------------------------------------------------------

def _build_gmm(samples: np.ndarray, n_components: int = 5) -> GaussianMixture:
    """Fit a K‑component full‑covariance GMM to a set of RGB samples."""
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", max_iter=100)
    gmm.fit(samples)
    return gmm


def _unary_costs(img: np.ndarray, fg_gmm: GaussianMixture, bg_gmm: GaussianMixture) -> tuple[np.ndarray, np.ndarray]:
    """Return negative log‑likelihood (data cost) for each pixel being FG / BG."""
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)  # (H*W, 3)
    fg_cost = -fg_gmm.score_samples(flat).reshape(h, w)
    bg_cost = -bg_gmm.score_samples(flat).reshape(h, w)
    return fg_cost, bg_cost


def grabcut_simple(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    iter_count: int = 5,
    lam: float = 50.0,
    sigma: float = 10.0,
) -> np.ndarray:
    """Run a pared‑down GrabCut.

    Parameters
    ----------
    img : np.ndarray (H, W, 3) float32 in [0,1]
    rect : (x, y, w, h) initial rectangle believed to contain the foreground
    iter_count : int  number of graph‑cut / GMM refinement iterations
    lam : float  smoothness weight
    sigma : float  colour similarity scale for smoothness term

    Returns
    -------
    mask : np.ndarray (H, W) uint8  — 1 = foreground, 0 = background
    """
    h, w, _ = img.shape

    # 0: BG, 1: FG, 2: PR_BG, 3: PR_FG   — OpenCV's original coding
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = rect
    mask[y : y + rh, x : x + rw] = 3  # probable FG inside rectangle
    mask[mask == 0] = 2  # probable BG elsewhere

    # pixel indices grid (shared across iterations)
    structure_horizontal = np.array([[0, 1]])
    structure_vertical = np.array([[1], [0]])

    for _ in range(iter_count):
        fg_samples = img[(mask == 1) | (mask == 3)].reshape(-1, 3)
        bg_samples = img[(mask == 0) | (mask == 2)].reshape(-1, 3)
        if fg_samples.size == 0 or bg_samples.size == 0:
            break  # degenerate state

        fg_gmm = _build_gmm(fg_samples)
        bg_gmm = _build_gmm(bg_samples)
        fg_cost, bg_cost = _unary_costs(img, fg_gmm, bg_gmm)

        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((h, w))

        print(nodeids.shape, left_w.shape, up_w.shape)

        # data term: link to source (FG) and sink (BG)
        g.add_grid_tedges(nodeids, fg_cost, bg_cost)

        # smoothness term — 4‑neighbourhood
        left_diff = np.sum((img[:, 1:] - img[:, :-1]) ** 2, axis=2)
        up_diff = np.sum((img[1:] - img[:-1]) ** 2, axis=2)
        left_w = lam * np.exp(-left_diff / (2 * sigma**2))
        up_w = lam * np.exp(-up_diff / (2 * sigma**2))

        g.add_grid_edges(nodeids[:, :-1], left_w, symmetric=True)
        g.add_grid_edges(nodeids[:-1, :], up_w, symmetric=True)

        g.maxflow()
        new_mask_fg = g.get_grid_segments(nodeids)  # True = BG reachable!? swap later
        mask = np.where(new_mask_fg, 0, 1).astype(np.uint8)  # 1 = FG, 0 = BG

        # Re‑label for next iteration: sure/unsure
        mask = np.where(mask == 1, 3, 2)
        # Optional: keep user certainty — this toy version assumes no extra strokes

    return (mask == 3).astype(np.uint8)  # back to 0/1 binary


# -----------------------------------------------------------------------------
#  CLI for quick experimentation
# -----------------------------------------------------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description="Minimal GrabCut re‑implementation (no cv2.grabCut)")
    p.add_argument("input", type=Path, help="input image path")
    p.add_argument("output", type=Path, help="output PNG path (RGBA)")
    p.add_argument("--rect", nargs=4, type=int, metavar=("x", "y", "w", "h"), required=True)
    p.add_argument("--iter", type=int, default=5, help="number of GrabCut iterations [5]")
    args = p.parse_args()

    img = io.imread(args.input)
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    
    print(f"img shape: {img.shape}, rect: {args.rect}")


    mask = grabcut_simple(img, tuple(args.rect), iter_count=args.iter)

    # write RGBA PNG
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack(((img * 255).astype(np.uint8), alpha))
    io.imsave(args.output, rgba)


if __name__ == "__main__":
    _cli()
