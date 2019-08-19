from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _get_interpolated_corner(corners_map, w, h, shape):
    w_fr = torch.floor(w)
    h_fr = torch.floor(h)
    w_ct = w_fr.long()
    h_ct = h_fr.long()
    res_w, res_h = 0, 0
    for iw,ih in [(0,0), (0,1), (1,0), (1,1)]:
        ids = (w_ct+iw + (h_ct+ih) * shape[1])
        _res_wh = _transpose_and_gather_feat(corners_map, w_ct+iw + (h_ct+ih) * shape[1]) * \
                         torch.abs((1-iw)-(w-w_fr)).unsqueeze(2) * torch.abs((1-ih)-(h-h_fr)).unsqueeze(2)
        res_w += _res_wh[:, :, 0:1]
        res_h += _res_wh[:, :, 1:2]
    return res_w, res_h

def corners2ctwh(corners):
    centers = torch.cat(
        [
            torch.mean(corners[:,:,:4], 2, keepdim=True),
            torch.mean(corners[:,:,4:], 2, keepdim=True)
        ], dim=2)
    wh = torch.cat(
        [
            2 * torch.mean(torch.abs(corners[:, :, :4] - centers[:, :, :1]), 2, keepdim=True),
            2 * torch.mean(torch.abs(corners[:, :, 4:] - centers[:, :, 1:]), 2, keepdim=True)
        ], dim=2)

    return centers, wh


def get_corners(corners_map, centers, wh, shape, method, opt):
    corners = torch.cat(
        [
            centers[:, :, 0:1] - wh[:, :, 0:1] / 2,
            centers[:, :, 0:1] - wh[:, :, 0:1] / 2,
            centers[:, :, 0:1] + wh[:, :, 0:1] / 2,
            centers[:, :, 0:1] + wh[:, :, 0:1] / 2,
            centers[:, :, 1:2] - wh[:, :, 1:2] / 2,
            centers[:, :, 1:2] + wh[:, :, 1:2] / 2,
            centers[:, :, 1:2] - wh[:, :, 1:2] / 2,
            centers[:, :, 1:2] + wh[:, :, 1:2] / 2
        ], dim=2
    )
    shape_h, shape_w = shape
    corners_w = torch.clamp(corners[:,:,:4], min=0, max=shape_w-1-1e-4)
    corners_h = torch.clamp(corners[:,:,4:], min=0, max=shape_h-1-1e-4)
    corners = torch.cat([corners_w, corners_h], 2)
    with torch.no_grad():
        corners = corners * 1
        wh = wh * 1
    if method == 'interpolation':
        res_corners_w = []
        res_corners_h = []
        for i in range(4):
            w, h = _get_interpolated_corner(corners_map[:,np.array([i,i+4], dtype=np.int32)],
                                            corners[:,:,i], corners[:,:,i+4], shape)
            if opt.normalized_corner:
                w = w * wh[:, :, 0:1]
                h = h * wh[:, :, 1:2]
            res_corners_w.append(w)
            res_corners_h.append(h)
        res_corners = res_corners_w
        res_corners.extend(res_corners_h)
        res_corners = torch.cat(res_corners, 2)
    else:
        raise NotImplemented

    return corners, res_corners

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)