
from unet_parts import *

from block import ResBlock
from block import GatedConv2d
from block import DilatedResBlock
from block import SpectralNorm


class AttentionNorm(nn.Module):

    def __init__(self, in_ch, in_ch1, sub_sample=3):
        super(AttentionNorm, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(in_ch, affine=True)
        self.conv = [nn.ReLU(inplace=True)]
        self.conv.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv = nn.Sequential(*self.conv)
        self.main = list()
        temp_ch = 32

        self.main.append(nn.Conv2d(in_ch1, temp_ch, kernel_size=3, stride=2, padding=1,bias=False))
        self.main.append(nn.BatchNorm2d(temp_ch, affine=True))
        self.main.append(nn.LeakyReLU(0.01, inplace=True))
        for _ in range(sub_sample - 1):
            self.main.append(nn.Conv2d(temp_ch, temp_ch * 2, kernel_size=3, stride=2, padding=1, bias=False))
            self.main.append(nn.BatchNorm2d(temp_ch * 2, affine=True))
            self.main.append(nn.LeakyReLU(0.01, inplace=True))
            temp_ch = temp_ch * 2

        self.main = nn.Sequential(*self.main)
        self.a = nn.Conv2d(temp_ch // 2, in_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.b = nn.Conv2d(temp_ch // 2, in_ch, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, feat, x):
        x = self.main(x)
        a = self.sigmoid(self.a(x))
        b = self.b(x)
        norm_feat = self.batch_norm(feat)

        return self.conv(a * norm_feat + b) + feat


# Free-Form Parsing Network
class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.res_layers = [ResBlock(1024 // factor, 1024 // factor) for _ in range(6)]
        self.res_layers = nn.Sequential(*self.res_layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.res_layers(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ParsingInpaintNet(nn.Module):

    def __init__(self):
        super(ParsingInpaintNet, self).__init__()

        self.att_norm1 = AttentionNorm(in_ch=1024, in_ch1=12, sub_sample=1)
        self.att_norm2 = AttentionNorm(in_ch=512, in_ch1=12, sub_sample=2)
        self.att_norm3 = AttentionNorm(in_ch=256, in_ch1=12, sub_sample=3)
        self.att_norm4 = AttentionNorm(in_ch=128, in_ch1=12, sub_sample=4)

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.en1 = Encoder(mode='seg_map')
        self.en2 = Encoder(mode='')
        self.res_layers = [DilatedResBlock(1024, 1024) for _ in range(6)]
        self.res_layers = nn.Sequential(*self.res_layers)
        self.to_rgb = list()
        self.to_rgb.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.to_rgb.append(nn.BatchNorm2d(64, affine=True))
        self.to_rgb.append(nn.LeakyReLU(0.01, inplace=True))
        self.to_rgb.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.to_rgb.append(nn.BatchNorm2d(32, affine=True))
        self.to_rgb.append(nn.LeakyReLU(0.01, inplace=True))
        self.to_rgb.append(nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False))
        self.to_rgb.append(nn.Tanh())
        self.to_rgb = nn.Sequential(*self.to_rgb)
        self.up_sample = nn.Upsample(scale_factor=2)

    def forward(self, seg_map, comp1, comp2):
        seg_feat = self.en1(seg_map)
        feat1 = self.en2(comp1)

        comb_feat = torch.cat((seg_feat, feat1), 1)
        comb_feat = self.res_layers(comb_feat)

        feat = self.att_norm1(comb_feat, comp2)
        feat = self.up_sample(feat)
        feat = self.conv1(feat)

        feat = self.att_norm2(feat, comp2)
        feat = self.up_sample(feat)
        feat = self.conv2(feat)

        feat = self.att_norm3(feat, comp2)
        feat = self.up_sample(feat)
        feat = self.conv3(feat)

        feat = self.att_norm4(feat, comp2)
        feat = self.up_sample(feat)

        return self.to_rgb(feat)


class Encoder(nn.Module):

    def __init__(self, mode='seg_map', n_layers=4):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.main = list()
        if mode == 'seg_map':
            Conv = nn.Conv2d
            in_ch = 3
        else:
            Conv = GatedConv2d
            in_ch = 12

        temp_ch = 32
        # 320 160 80 40 20
        # 512 256 128 64 32
        # 32 64 128 256 512

        self.main.append(Conv(in_ch, temp_ch, kernel_size=3, stride=2, padding=1))
        self.main.append(nn.BatchNorm2d(temp_ch))
        self.main.append(nn.LeakyReLU(0.01, inplace=True))
        for _ in range(self.n_layers - 1):
            self.main.append(Conv(temp_ch, temp_ch * 2, kernel_size=3, stride=2, padding=1))
            self.main.append(nn.BatchNorm2d(temp_ch * 2))
            self.main.append(nn.LeakyReLU(0.01, inplace=True))
            temp_ch = temp_ch * 2
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, base_channel=32, nlayers=4, LR=0.01):
        super(Discriminator, self).__init__()

        layers = list()

        layers.append(SpectralNorm(nn.Conv2d(3, base_channel, kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(nn.InstanceNorm2d(base_channel, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        for _ in range(nlayers - 2):
            layers.append(SpectralNorm(nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False)))
            layers.append(nn.InstanceNorm2d(base_channel * 2, affine=True))
            layers.append(nn.LeakyReLU(LR, inplace=True))
            base_channel = base_channel * 2

        layers.append(nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=2, padding=1, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

