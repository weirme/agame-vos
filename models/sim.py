import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.model_utils import *


class SIM(nn.Module):
    def __init__(self, local_size, chileren_cfg):
        super().__init__()
        self.local_size = local_size
        self.merge_layers = nn.Sequential(OrderedDict([('layer{:02d}'.format(idx), SUBMODS[child_cfg[0]](*child_cfg[1])) for idx, child_cfg in enumerate(chileren_cfg)]))

    def get_init_state(self, input_feats, given_seg):
        dynmod_state = {'init_feats': input_feats, 'init_seg': given_seg, 'prev_feats': input_feats, 'prev_seg': given_seg}
        return dynmod_state

    def update(self, input_feats, predicted_segmentation, state):
        dynmod_state = state['dynmod']
        dynmod_state['prev_feats'] = input_feats
        dynmod_state['prev_seg'] = predicted_segmentation
        return dynmod_state

    def get_global_map(self, init_feats, init_seg, cur_feats):
        init_feats = init_feats['embed']
        cur_feats = cur_feats['embed']
        init_feats = F.avg_pool2d(init_feats, 4)
        init_seg = F.avg_pool2d(init_seg, 4)
        nbatch, nchannel, h, w = init_feats.size()
        similarity_maps = []
        for b in range(nbatch):
            init_feats_i = init_feats[b].view(-1, nchannel, 1, 1)
            cur_feats_i = cur_feats[b].unsqueeze(0)
            sim_map = F.conv2d(cur_feats_i, init_feats_i) # (1, h0*w0, h, w)
            similarity_maps.append(sim_map)
        similarity_maps = torch.stack(similarity_maps) # (nbatch, 1, h0*w0, h, w)
        init_seg = init_seg.view(nbatch, 2, -1, 1, 1) 
        similarity_maps = similarity_maps * init_seg # (nbatch, 2, h0*w0, h, w)
        similarity_maps = similarity_maps.max(dim=2)[0] # (nbatch, 2, h, w)
        return similarity_maps

    def get_local_map(self, prev_feats, prev_seg, cur_feats, window_size):
        xcorr = Correlation(pad_size=window_size, kernel_size=1, max_displacement=window_size, stride1=1, stride2=1)
        prev_feats = prev_feats['embed']
        cur_feats = cur_feats['embed']
        bg_seg = prev_seg[:, 0].unsqueeze(1)
        fg_seg = prev_seg[:, 1].unsqueeze(1)
        bg_feats = prev_feats * bg_seg
        fg_feats = prev_feats * fg_seg
        bg_map = xcorr(cur_feats, bg_feats).max(dim=1)[0].unsqueeze(1)
        fg_map = xcorr(cur_feats, fg_feats).max(dim=1)[0].unsqueeze(1)
        return torch.cat([bg_map, fg_map], dim=1)

    def forward(self, input_feats, state):
        dynmod_state = state['dynmod']
        cur_feats = input_feats 
        init_feats = dynmod_state['init_feats']
        prev_feats = dynmod_state['prev_feats'] 
        init_seg = dynmod_state['init_seg'] 
        prev_seg = dynmod_state['prev_seg'] 
        
        global_map = self.get_global_map(init_feats, init_seg, cur_feats)
        local_map = self.get_local_map(prev_feats, prev_seg, cur_feats, self.local_size)
        
        h = torch.cat([cur_feats['decode'], global_map, init_seg, local_map, prev_seg], dim=1)

        h = self.merge_layers(h)

        return h, dynmod_state