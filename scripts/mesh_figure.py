"""Unchanged camera and shell rendering used by the experimental overview."""
import numpy as np
import matplotlib
from matplotlib.collections import PolyCollection

LIGHT = np.array([-.45, -.75, .49])/np.linalg.norm([-.45, -.75, .49])

def _camera(elev, azim):
    """Unit vector from the scene to the camera, with the image basis."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    cam = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    right = np.cross(-cam, [0., 0., 1.])
    right /= np.linalg.norm(right)
    return np.vstack([right, np.cross(right, -cam)]), cam


def _surface(ax, basis, cam, nodes, quads, tint, alpha=1., edge=None, lw=.12):
    """Shaded quads drawn back to front, so later markers sit on the surface."""
    v = nodes[quads]
    nrm = np.cross(v[:, 1]-v[:, 0], v[:, 3]-v[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    face = np.clip(np.array(matplotlib.colors.to_rgb(tint))[None]
                   * (.44+.56*abs(nrm@LIGHT))[:, None], 0, 1)
    order = np.argsort(v.mean(axis=1)@cam)
    flat = (v.reshape(-1, 3)@basis.T).reshape(len(v), 4, 2)
    ax.add_collection(PolyCollection(
        flat[order], facecolors=face[order], alpha=alpha, linewidths=lw,
        edgecolors=(face[order] if edge is None else edge)))

