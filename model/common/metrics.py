"""Segmentation metrics, single-sourced.

Two backends for the same ideas, because callers differ:

* **torch** (``dice_score``, ``boundary_f1``) — used inside training/eval loops on
  GPU tensors. Tolerant boundary-F1 dilates via ``max_pool2d`` (±``tol`` px).
* **numpy** (``dice_np``, ``boundary_f1_np``) — used by offline analysis scripts on
  CPU arrays. Tolerant boundary-F1 dilates via ``scipy.ndimage.binary_dilation``.

The two boundary-F1 flavors are numerically close but not bit-identical (pooling vs.
iterated 1-px dilation); each backend reproduces exactly what its callers used before.
"""
import torch.nn.functional as F


def _as_4d(t):
    """Convert to an (N, C, H, W) view for max_pool2d. Sums are global, so the
    exact leading shape doesn't affect the scalar result."""
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return t


def dice_score(prob, target, thr=0.5, eps=1e-8):
    """Soft-thresholded Dice over all elements. Returns a python float."""
    pb = (prob > thr).float()
    g = (target > 0.5).float()
    inter = (pb * g).sum()
    return (2 * inter / (pb.sum() + g.sum() + eps)).item()


def boundary_f1(prob, target, thr=0.5, tol=2, eps=1e-8):
    """Tolerant boundary-F1 (±``tol`` px) on torch tensors. Returns a python float.

    A predicted pixel counts as matched if a ground-truth pixel lies within ``tol``
    px (and vice versa), implemented by dilating each mask with a ``2*tol+1`` max-pool.
    Accepts any tensor of rank <= 4 with trailing (H, W) dims.
    """
    pb = _as_4d((prob > thr).float())
    g = _as_4d((target > 0.5).float())
    k = 2 * tol + 1
    gd = F.max_pool2d(g, k, 1, tol)
    pd = F.max_pool2d(pb, k, 1, tol)
    bp = (pb * gd).sum() / (pb.sum() + eps)   # precision: preds near a GT pixel
    br = (g * pd).sum() / (g.sum() + eps)     # recall: GT near a predicted pixel
    return (2 * bp * br / (bp + br + eps)).item()


def dice_np(a, b, empty_score=1.0):
    """Dice on boolean/float numpy arrays. Both-empty returns ``empty_score``."""
    a = a > 0.5
    b = b > 0.5
    s = a.sum() + b.sum()
    if s == 0:
        return empty_score
    return 2.0 * (a & b).sum() / s


def boundary_f1_np(pred, gt, tol=2):
    """Tolerant boundary-F1 (±``tol`` px) on numpy arrays via iterated 1-px dilation.

    Matches the offline analysis metric: empty-vs-empty -> 1.0, one-empty -> 0.0.
    """
    from scipy import ndimage

    pred = pred > 0.5
    gt = gt > 0.5
    st = ndimage.generate_binary_structure(2, 1)
    gt_d = ndimage.binary_dilation(gt, st, iterations=tol)
    pred_d = ndimage.binary_dilation(pred, st, iterations=tol)
    if pred.sum() == 0 or gt.sum() == 0:
        return 1.0 if (pred.sum() == 0 and gt.sum() == 0) else 0.0
    prec = (pred & gt_d).sum() / pred.sum()
    rec = (gt & pred_d).sum() / gt.sum()
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
