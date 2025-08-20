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
        in_ch = 8
        out_ch = 8

        n1 = 32
        filters = [n1 * 1, n1 * 2, n1 * 3, n1 * 4, n1 * 5]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.Conv0 = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[0]))
        self.Conv1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], 3, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[0]))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[1]))
        self.Conv3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], 3, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[1]))
        self.Avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.line1 = nn.Sequential(nn.Linear(64, 512),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(512))
        self.line2 = nn.Sequential(nn.Linear(512, 512),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm1d(512))
        self.line3 = nn.Linear(512, 7)


    def forward(self, x):

        e0 = self.Conv0(x)
        e0 = self.Maxpool(e0)
        e1 = self.Conv1(e0)
        # print(e2.shape)
        e1 = self.Maxpool(e1)
        e2 = self.Conv2(e1)

        e2 = self.Maxpool(e2)
        # pe1 = self.pe1(cf1)
        e3 = self.Conv3(e2)

        # pe2 = self.pe2(cf2)
        # print(e3.shape)
        e4 = self.Avgpool(e3)

        # print(e4.shape)

        d1 = torch.flatten(e4, 1)
        # print(d1.shape)
        l1 = self.line1(d1)  # 16 2 2
        # print(l1.shape)
        l2 = self.line2(l1)  # 16 2 2
        out = self.line3(l2)  # 16 2 2
        # print(out.shape)

        return out


if __name__ == '__main__':
    x = torch.randn(2, 8, 128, 128)
    net = U_Net()
    print(net(x).shape)

