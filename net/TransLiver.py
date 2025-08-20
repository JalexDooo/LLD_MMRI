import logging

import timm
import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
# from wandb.wandb_torch import torch

from .vit_timm import create_mbt


def generate_model_mbt(model_type="mbt", model_scale="base", model_size=224, patch_size=16,
                       no_cuda=False, is_multi=False, pretrain_path=None, nb_class=7, drop_out=0.0,
                       phase_num=1, bottleneck_n=4, backbone="vit", in_channel=8):
    assert model_type in [
        'mbt'
    ]
    assert model_scale in [
        'base'
    ]
    assert model_size in [224]
    assert patch_size in [16]

    model_name = model_type + '_' + model_scale + '_phase' + str(phase_num) + '_bottleneck' + str(bottleneck_n) + \
                     '_' + backbone

    # model = timm.create_model(model_name, pretrained=False if pretrain_path is None else True,
    #                           pretrain_path=pretrain_path, num_classes=nb_class,
    #                           in_chans=in_channel, drop_rate=drop_out)
    model = create_mbt(model_name, pretrained=False if pretrain_path is None else True,
                       pretrain_path=pretrain_path, num_classes=nb_class,
                       in_chans=in_channel, drop_rate=drop_out)

    return model


if __name__ == '__main__':
    x = torch.randn(8, 8, 224, 224).unsqueeze(0)
    print(x.shape)
    net = generate_model_mbt()
    print(net(x).shape)
