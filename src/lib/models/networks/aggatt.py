#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from .DCNv2.dcn_v2 import dcn_v2_conv



class AggAtt(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1,
                 deformable_groups=1, final_channels=2, mode='v1'):
        super(AggAtt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.mode = mode
        if mode == 'v1':
            channels_ = 7
        elif mode == 'v2':
            channels_ = 11
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(out_channels, final_channels,
                                 kernel_size=1, stride=1,
                                 padding=0, bias=True)

        self.reset_parameters()
        self.init_offset()


        dcn_pad = 1
        dcn_base = np.arange(-dcn_pad, dcn_pad + 1).astype(np.float32)
        dcn_kernel = 3
        dcn_base_y = np.repeat(dcn_base, dcn_kernel)
        dcn_base_x = np.tile(dcn_base, dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1,-1,1,1)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
        self.conv1x1.weight.data.zero_()
        self.conv1x1.bias.data.zero_()

    def forward(self, input, wh):
        w = wh[:,:1,:,:]

        h = wh[:,1:2,:,:]
        w = 0.01 * w + w.detach() * 0.99
        h = 0.01 * h + h.detach() * 0.99
        po = torch.zeros([wh.shape[0], 1, wh.shape[2], wh.shape[3]]).cuda()
        po1 = torch.ones([wh.shape[0], 1, wh.shape[2], wh.shape[3]]).cuda()
        offset_mask = self.conv_offset_mask(input)
        offset = offset_mask[:,:2]
        mask = torch.sigmoid(offset_mask[:,2:])
        if self.mode == 'v1':
            mask = torch.cat([mask[:,0:1], po, mask[:,1:2],\
                              po,mask[:,2:3],po,\
                              mask[:,3:4],po,mask[:,4:5]],1)
            offset = torch.cat([-h / 2, -w / 2, po, po, -h / 2, w / 2, \
                                po, po, offset[:, :1], offset[:, 1:], po, po, \
                                h / 2, -w / 2, h / 2, po, po, w / 2], 1) - self.dcn_base_offset.cuda()
        elif self.mode == 'v2':
            mask = torch.cat([mask[:,0:1], mask[:,1:2], mask[:,2:3], \
                              mask[:,3:4],mask[:,4:5],mask[:,5:6], \
                              mask[:,6:7],mask[:,7:8],mask[:,8:9]],1)

            offset = torch.cat([-h / 2, -w / 2, -h / 2,po, -h / 2, w / 2, \
                              po, -w / 2, offset[:,:1], offset[:,1:], po, w / 2, \
                              h / 2, -w / 2, h / 2, po, h / 2, w / 2],1) - self.dcn_base_offset.cuda()

        dcn_feat = dcn_v2_conv(input, offset, mask,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)
        dcn_feat_relu = self.relu(dcn_feat)
        return self.conv1x1(dcn_feat_relu)