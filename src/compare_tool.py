import os
import pickle
import argparse
import numpy as np
from numpy import array as npa

from IPython import embed

def create_argparse():
    parser = argparse.ArgumentParser(description='compare')
    parser.add_argument('src', type=str)
    parser.add_argument('dst', type=str)
    parser.add_argument('--box_fracs_scales', type=str, default='0,8,16,32,64,128,256,512',
                             help='inner-box attribution for different scale')

    return parser

if __name__ == '__main__':
    args_parser = create_argparse()
    opt = args_parser.parse_args()
    opt.box_fracs_scales = list(map(float, opt.box_fracs_scales.split(',')))

    with open(os.path.join('../exp', opt.src, 'box_fracs.pkl'), 'rb') as f:
        f_src_list = pickle.load(f)
        s_src = pickle.load(f)

    with open(os.path.join('../exp', opt.dst, 'box_fracs.pkl'), 'rb') as f:
        f_dst_list = pickle.load(f)
        s_dst = pickle.load(f)

    opt.box_fracs_scales.append(1e9)
    ratio_list = [0.5, 0.75, 1.0, 1.25, 1.5]
    for idx, box_size in enumerate(opt.box_fracs_scales[:-1]):
        for k in range(len(ratio_list)):
            ratio = ratio_list[k]
            pos, neg = 0, 0
            for i in range(len(f_src_list[0])):
                try:
                    diff = np.array(f_src_list[i][k]) - np.array(f_dst_list[i][k])
                    pos += ((diff > 0) * (npa(s_dst[i]).mean(1) > box_size) * (npa(s_dst[i]).mean(1) <= opt.box_fracs_scales[idx+1])).sum()
                    neg += ((diff < 0) * (npa(s_src[i]).mean(1) > box_size) * (npa(s_src[i]).mean(1) <= opt.box_fracs_scales[idx+1])).sum()
                except Exception as e:
                    continue
            print('size@{}~{}|ratio@{} - > : {}, < : {}'.format(box_size, opt.box_fracs_scales[idx+1], ratio, pos, neg))
