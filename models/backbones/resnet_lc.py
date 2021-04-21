from .resnet import *
import torch.nn as nn
from models.model_utils import *


class ResNetS16LC(nn.Module):
    def __init__(self, finetune_layers, **kwargs):
        super().__init__()
        self.resnet = resnet101(pretrained=True, **kwargs)
        self.finetune_layers = finetune_layers

        self.resnet.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
        for module_name in finetune_layers:
            getattr(self.resnet, module_name).train(True)
            getattr(self.resnet, module_name).requires_grad = True
            for param in getattr(self.resnet, module_name).parameters():
                param.requires_grad = True

        self.aspp = ASPP()
        self.decoder = Decoder(256)
        self.embedding = SeparableConv2d(256, 128)

        self.lcm1 = LayerCascadeModule(256, 64, 512, 64, 256)
        self.lcm2 = LayerCascadeModule(512, 256, 1024, 64, 256)
        self.lcm3 = LayerCascadeModule(1024, 512, 2048, 64, 256)
        self.lcm4 = LayerCascadeModule(2048, 1024, None, 64, 256)

    def get_features(self, x):
        f = self.resnet.get_features(x)
        decode = self.decoder(self.aspp(f[3]), f[1])
        embed = self.embedding(decode)
        feats = {}
        feats['decode'] = decode
        feats['embed'] = embed
        l1 = self.lcm1(f[1], f[0], f[2])
        feats['lc1'] = l1
        l2 = self.lcm2(f[2], f[1], f[3])
        feats['lc2'] = l2
        l3 = self.lcm3(f[3], f[2], f[4])
        feats['lc3'] = l3
        l4 = self.lcm4(f[4], f[3], None)
        feats['lc4'] = l4

        return feats


def resnet101lc(finetune_layers=(), **kwargs):
    model = ResNetS16LC(finetune_layers, **kwargs)
    return model