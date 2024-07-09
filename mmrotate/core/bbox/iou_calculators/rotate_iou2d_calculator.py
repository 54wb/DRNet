# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import shapely.geometry as shgeo
from mmcv.ops import box_iou_rotated

from .builder import ROTATED_IOU_CALCULATORS


@ROTATED_IOU_CALCULATORS.register_module()
class RBboxOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode='iou',
                 is_aligned=False,
                 version='oc'):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, or shape (m, 6) in
                 <cx, cy, w, h, a2, score> format.
            bboxes2 (torch.Tensor): bboxes have shape (m, 5) in
                <cx, cy, w, h, a> format, shape (m, 6) in
                 <cx, cy, w, h, a, score> format, or be empty.
                 If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            version (str, optional): Angle representations. Defaults to 'oc'.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return rbbox_overlaps(bboxes1.contiguous(), bboxes2.contiguous(), mode,
                              is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def rbbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (torch.Tensor): shape (B, m, 5) in <cx, cy, w, h, a> format
            or empty.
        bboxes2 (torch.Tensor): shape (B, n, 5) in <cx, cy, w, h, a> format
            or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof']
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert (bboxes1.size(-1) == 5 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 5 or bboxes2.size(0) == 0)

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)



def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """caculator the overlap between two bboxes no matter its type"""
    
    """
    Args:
        bboxes1 (numpy.array):shape(n,8)  (x1,y1,x2,y2,x3,y3,x4,y4)
        bboxes2 (numpy.array):shape(1,4)  (xlt,ylt,xrb,yrb)

    """
    assert mode in ['iou', 'iof']
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return np.zeros((rows, 1), dtype=np.float32) \
                if is_aligned else np.zeros((rows, cols), dtype=np.float32)

    hbboxes1 = poly2xyxy(bboxes1)
    hbboxes2 = bboxes2
    if not is_aligned:
        hbboxes1 = hbboxes1[:, None, :]
    lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
    rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]
    
    polys1 = bboxes1
    polys2 = xyxy2poly(bboxes2)
    sg_polys1 = [shgeo.Polygon(p) for p in polys1.reshape(rows, -1, 2)]
    sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]

    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        try:
            overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
        except:
            overlaps[p] = 0

    if mode == 'iou':
        unions = np.zeros(h_overlaps.shape, dtype=np.float32)
        for p in zip(*np.nonzero(h_overlaps)):
            unions[p] = sg_polys1[p[0]].union(sg_polys2[p[-1]]).area
    else:
        unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
        if not is_aligned:
            unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs

def poly2xyxy(polys):
    shape = polys.shape
    polys = polys.reshape(*shape[:-1],shape[-1]//2,2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)

def xyxy2poly(hbboxes):
    l, t, r, b = [hbboxes[..., i] for i in range(4)]
    return np.stack([l, t, r, t, r, b, l, b], axis=-1)