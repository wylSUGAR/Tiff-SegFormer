# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, TiffTransBlock, TiffTransBlock1, TiffTransBlock2


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=2, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*8,
            c2=embedding_dim,
            k=1,
        )

        # wyl ---------- 加上通道注意力模块
        # self.eca_block = ECABlock(channels=embedding_dim*8)
        # wyl ---------- 加上CBAMBlock机制：通道和空间注意力机制
        self.cbam_block = CBAMBlock(channels=embedding_dim * 8)


        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs, tiffs):
        c1, c2, c3, c4 = inputs

        # -----------------wyl
        t1, t2, t3, t4 = tiffs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape


        # _c4 = self.linear_c4(torch.cat((c4, t4), dim=1)).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c3 = self.linear_c3(torch.cat((c3, t3), dim=1)).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c2 = self.linear_c2(torch.cat((c2, t2), dim=1)).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c1 = self.linear_c1(torch.cat((c1, t1), dim=1)).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # ----------------------- wyl: tiff
        _t4 = self.linear_c4(t4).permute(0, 2, 1).reshape(n, -1, t4.shape[2], t4.shape[3])
        _t4 = F.interpolate(_t4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _t3 = self.linear_c3(t3).permute(0, 2, 1).reshape(n, -1, t3.shape[2], t3.shape[3])
        _t3 = F.interpolate(_t3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _t2 = self.linear_c2(t2).permute(0, 2, 1).reshape(n, -1, t2.shape[2], t2.shape[3])
        _t2 = F.interpolate(_t2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _t1 = self.linear_c1(t1).permute(0, 2, 1).reshape(n, -1, t1.shape[2], t1.shape[3])
        # -----------------------



        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        concat_channels = torch.cat([_c4, _c3, _c2, _c1, _t4, _t3, _t2, _t1], dim=1)
        # wyl------------------ concat 通道后，加上通道注意力机制，查看效果
        cbam__channels = self.cbam_block(concat_channels)

        _c = self.linear_fuse(cbam__channels)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


##------------wyl :  将可见光和 tiff(温度) channels 维度concat后， 加上通道注意力机制，给各通道加上权重
class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


##------------wyl :  将可见光和 tiff(温度) channels 维度concat后， 加上空间注意力机制，给各通道加上权重
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        v = torch.cat([avg_out, max_out], dim=1)
        v = self.conv1(v)
        v = self.sigmoid(v)
        return x * v


##------------wyl :  将可见光和 tiff(温度) channels 维度concat后， 加上通道注意力机制和空间注意力机制，给各通道加上权重
class CBAMBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channelattention = ECABlock(channels)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x



#--------wyl: 将RGB和tiff图像进行cancat操作，然后进行一个卷积，使得通道变为3层，其余不变；  该方法测试后效果不显著，暂作废
class MixVisionTiff(nn.Module):
    def __init__(self, inut_dim=4, output_dim=3, k=1, s=1, p=0, g=1, act=True):
        super(MixVisionTiff, self).__init__()
        self.conv = nn.Conv2d(inut_dim, output_dim, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(output_dim, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x, t):
        x_temp = torch.cat([x, t], dim=1)
        return self.act(self.conv(x_temp))



class SegFormer(nn.Module):
    def __init__(self, num_classes = 2, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

        # -----wyl
        self.tiffTransBlock = TiffTransBlock(pretrained)


    def forward(self, inputs, tiffs):

        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)

        # --------wyl
        t = self.tiffTransBlock.forward(tiffs)

        x = self.decode_head.forward(x, t)
        
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
