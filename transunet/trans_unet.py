import torch.nn as nn
from einops import rearrange

from .bottleneck_layer import Bottleneck
from .decoder import Up, SignleConv
from ..vit import ViT
from .weighted_block import WeightedBlock

class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=4,
                 vit_dim_linear_mhsa_block=1024,
                 vit_transformer=None,
                 vit_channels = None
                 ):
        """
        My reimplementation of TransUnet based on the paper:
        https://arxiv.org/abs/2102.04306
        Badly written, many details missing and significantly differently
        from the authors official implementation (super messy code also :P ).
        My implementation doesnt match 100 the authors code.
        Basically I wanted to see the logic with vit and resnet backbone for
        shaping a unet model with long skip connections.

        Args:
            img_dim: the img dimension
            in_channels: channels of the input
            classes: desired segmentation classes
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
            vit_transformer: pass your own version of vit
            vit_channels: the channels of your pretrained vit. default is 128*8
        """
        super().__init__()
        self.inplanes = 128
        vit_channels = self.inplanes * 8 if vit_channels is None else vit_channels

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.
        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        self.img_dim_vit = img_dim // 16

        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # encoder channels
                       patch_dim=1,
                       dim=vit_channels,  # vit out channels for decoding
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        self.vit_conv = SignleConv(in_ch=vit_channels, out_ch=512)

        self.weight1 = WeightedBlock(256, 16)
        self.weight2 = WeightedBlock(128, 16)
        self.weight3 = WeightedBlock(64,16)
        self.weight4 = WeightedBlock(16,16)

        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1 = Up(1024, 256)
        self.dec2 = Up(512, 128)
        self.dec3 = Up(256, 64)
        self.dec4 = Up(64, 16)
        self.conv1x1 = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        # ResNet 50-like encoder
        x2 = self.init_conv(x)  # 128,64,64
        x4 = self.conv1(x2)  # 256,32,32
        x8 = self.conv2(x4)  # 512,16,16
        x16 = self.conv3(x8)  # 1024,8,8
        y = self.vit(x16)
        y = rearrange(y, 'b (x y) dim -> b dim x y ', x=self.img_dim_vit, y=self.img_dim_vit)
        y = self.vit_conv(y)
        y1 = self.dec1(y, x8)  # 256,16,16
        y2 = self.dec2(y1, x4) # 128
        y3 = self.dec3(y2, x2) # 64
        y = self.dec4(y3) # 16

        w1 = self.weight1(y1)
        w1 = self.up_1(w1)
        w2 = self.weight2(y2) + w1
        w2 = self.up_2(w2)
        w3 = self.weight3(y3) + w2
        w3 = self.up_3(w3)
        w4 = self.weight4(y) + w3
        
        return self.conv1x1(w4)
