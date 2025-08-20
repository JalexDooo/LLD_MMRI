import math
import time
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch


class conv_block_(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 1, kernel_size=[3, 3, 3], stride=[2, 1, 1], padding=1)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv_block(out_ch, out_ch, out_ch)

            # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, 0, :, :, :]
        x = self.conv2(x)
        return x


class conv_block_3d(nn.Module):
    # conv + conv
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
            # LayerNormChannel(1, eps=1e-6),
            # nn.LayerNorm([out_ch, 1, 512, 512]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]).shape)
        x = self.conv(x.reshape(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))
        # print(x.shape)
        return x.reshape(x.shape[0], 128, x.shape[3], x.shape[4])


class PatchEmbed(nn.Module):

    def __init__(self, kernel=3, stride=2, padding=1, in_ch=1, out_ch=768, norm_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        # self.norm = LayerNormChannel(out_ch, eps=1e-6)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):

    def __init__(self, in_ch, hidden_ch=None,
                 out_ch=None, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or in_ch
        # self.fc1 = nn.Sequential(
        #     nn.Conv2d(in_ch, hidden_ch, kernel_size=1),
        #     nn.BatchNorm2d(hidden_ch),
        #     nn.ReLU(inplace=True)
        # )
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_ch, out_ch, 1)
        # self.fc2 = nn.Sequential(
        #     nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )
        # self.drop = nn.Dropout(drop)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)

        # x = self.drop(x)
        return x


class CnnFormerBlock(nn.Module):

    def __init__(self, in_ch, mlp_ratio=4., k=64,
                 drop=0., use_layer_scale=False, layer_scale_init_value=1e-5):

        super().__init__()

        # self.token_mixer = CoTAttention(dim=in_ch, kernel_size=3)
        # self.token_mixer = SequentialPolarizedSelfAttention(channel=in_ch)
        # self.token_mixer = CrissCrossAttention(in_ch)
        # self.conv_1 = nn.Conv2d(in_channels=dim, out_channels=96, stride=1, kernel_size=3, padding=1)

        self.token_mixer = conv_block(in_ch=in_ch, out_ch=in_ch, out_ch1=in_ch)
        # self.token_mixer = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 1, 1),
        #     nn.BatchNorm2d(in_ch),
        #     nn.Sigmoid())
        # self.norm2 = norm_layer(in_ch)
        mlp_hidden_dim = int(in_ch * mlp_ratio)
        self.mlp = Mlp(in_ch=in_ch, hidden_ch=mlp_hidden_dim, drop=drop)

        # The following two techniques are useful to train deep CnnFormers.

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_ch)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_ch)), requires_grad=True)

    def forward(self, x):

        if self.use_layer_scale:
            # print(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).shape)
            # print(self.token_mixer(x).shape)
            x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(x)
            x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        else:
            x = x + self.token_mixer(x)
            x = x + self.mlp(x)
        return x


class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch, out_ch1, kernel=3, stride=1, padding=1, stride1=1, padding1=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch1, kernel_size=3, stride=stride1, padding=padding1, bias=True),
            nn.BatchNorm2d(out_ch1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def cat(x, y):
    b, c, l, w = x.shape
    return torch.cat((x, y), dim=2).reshape(b, 1, c * 2, l, w)


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self):
        super(U_Net, self).__init__()
        in_ch = 5
        out_ch = 8

        n1 = 64
        filters = [n1 * 2, n1 * 3, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool1 = PatchEmbed(in_ch=filters[0], out_ch=filters[0])
        # self.Maxpool2 = PatchEmbed(in_ch=filters[1], out_ch=filters[1])
        # self.Maxpool3 = PatchEmbed(in_ch=filters[2], out_ch=filters[2])
        # self.Maxpool4 = PatchEmbed(in_ch=filters[3], out_ch=filters[3])

        # self.Conv0 = conv_block_3d(1, 1, 3, 1, [0, 1, 1])
        # self.Conv1 = conv_block(1, filters[0])
        self.Conv0 = conv_block_3d(1, 16, [5, 3, 3], [5, 1, 1], [0, 1, 1])
        self.Conv1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, 1, 1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True))
        self.Conv2 = conv_block(filters[0], filters[1], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4], filters[4])

        # self.cf0 = CnnFormerBlock(in_ch=filters[0], k=int(filters[0] / 2))
        # self.cf1 = CnnFormerBlock(in_ch=filters[1], k=int(filters[1] / 8 * 3))
        # self.cf2 = CnnFormerBlock(in_ch=filters[2], k=int(filters[2] / 4))
        # self.cf3 = CnnFormerBlock(in_ch=filters[3], k=int(filters[3] / 8))
        # self.cf4 = CnnFormerBlock(in_ch=filters[4])
        # self.Con_cf4 = conv_block(filters[4], filters[4], filters[3])
        # self.pe0 = PatchEmbed(in_ch=filters[0], out_ch=filters[1])
        # self.pe1 = PatchEmbed(in_ch=filters[1], out_ch=filters[2])
        # self.pe2 = PatchEmbed(in_ch=filters[2], out_ch=filters[3])
        # self.pe3 = PatchEmbed(in_ch=filters[3], out_ch=filters[4])

        # self.Up4 = conv_block(filters[3], filters[4], filters[4], 5, 2, 2, 1, 1)
        self.Up_conv4 = conv_block(filters[4], filters[3], filters[3])

        # self.Up3 = conv_block(filters[2], filters[3], filters[4], 3, 2, 1, 2, 1)
        self.Up_conv3 = conv_block(filters[3], filters[2], filters[2], 3, 2, 1, 1, 1)

        # self.Up2 = conv_block(filters[1], filters[2], filters[4], 3, 2, 1, 2, 1)
        self.Up_conv2 = conv_block(filters[2], filters[1], filters[1])

        # self.Up1 = nn.Sequential(
        #     conv_block(filters[0], filters[2], filters[3], 3, 2, 1, 2, 1),
        #     nn.Conv2d(filters[3], filters[4], 3, 2, 1, 1, 1),
        #     nn.BatchNorm2d(filters[4]),
        #     nn.ReLU(inplace=True)
        # )

        self.Up_conv1 = conv_block(filters[1], filters[0], 24)

        # self.Conv = conv_block_3d(1, out_ch, kernel=[3, 3, 3], stride=1, padding=1)
        # self.Out = nn.Conv3d(filters[0], out_ch, kernel_size=[3, 1, 1], stride=1, padding=0)
        # self.Conv = nn.Sequential(
        #     nn.Conv2d(filters[0], 11, 1),
        #     nn.BatchNorm2d(11),
        #     nn.ReLU(inplace=True))
        # self.Out = nn.Conv3d(1, 1, kernel_size=[5, 1, 1], stride=[3, 1, 1], padding=0)

        self.Conv = nn.Linear(384, 7)


    def forward(self, x):
        # e1 = self.Conv0(torch.unsqueeze(x, dim=1))
        # e1 = self.Conv1(e1)
        e1 = self.Conv0(x)

        # cf0 = self.cf0(e1)
        e1 = self.Conv1(e1)
        # print(e2.shape)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        # print(e2.shape)
        # pe0 = self.pe0(cf0)
        # cf1 = self.cf1(e2)

        e3 = self.Maxpool(e2)
        # pe1 = self.pe1(cf1)
        e3 = self.Conv3(e3)
        # cf2 = self.cf2(pe1 + e3)

        e4 = self.Maxpool(e3)
        # pe2 = self.pe2(cf2)
        e4 = self.Conv4(e4)
        # cf3 = self.cf3(pe2 + e4)
        # print(cf3.shape)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)
        # pe3 = self.pe3(cf3)
        # cf4 = self.cf4(pe3 + e5)  # 1024 32 32
        #cf4 = self.Con_cf4(cf4)
        # print(cf4.shape)
        # cf3 = self.Up4(cf3)  # 512 64 64 -> 1024 32 32
        # cf2 = self.Up3(cf2)  # 256 128 128 -> 1024 32 32
        # cf1 = self.Up2(self.Maxpool(cf1))  # 128 256 256 -> 1024 32 32
        # e1 = self.Up1(self.Maxpool(e1))  # 64 512 512 -> 1024 32 32
        # print(cf3.shape, cf2.shape, cf1.shape, e1.shape)

        # d4 = torch.cat((cf4, cf3, cf2, cf1, e1), dim=1)

        d4 = self.Up_conv4(e5)  # 256 32 32
        # print(d4.shape)
        d3 = self.Up_conv3(d4)  # 128 16 16
        # print(d3.shape)
        d2 = self.Up_conv2(d3)  # 64 8 8
        # print(d2.shape)
        d1 = self.Up_conv1(d2)  # 32 4 4
        # print(d1[1:].shape)
        # print(torch.flatten(d1).shape)
        d1 = torch.flatten(d1, 1)
        # print(d1.shape)
        out = self.Conv(d1)  # 16 2 2
        # print(out.shape)
        # out = self.Out(out).permute(0, 2, 1, 3, 4)
        # out = self.Out(torch.unsqueeze(out, dim=1))
        # print(out.shape)
        #
        # out = self.active(out)
        return out


if __name__ == '__main__':
    x = torch.randn(2, 40, 128, 128)
    net = U_Net()

    time1 = time.time()
    print(net(x).shape)
    time2 = time.time()
    print(time2 - time1)
    # print(np.sum([p.numel() for p in net.parameters()]).item())
    # flops = FlopCountAnalysis(net, (x,))
    # print(flops.total())
