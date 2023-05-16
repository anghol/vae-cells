import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_corresponding):
        x = self.up(x)
        x = torch.cat((x_corresponding, x), dim=1)
        x = self.conv(x)
        return x



class UNet(nn.Module):
    def __init__(self, resolution: tuple, in_channels: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels

        self.input_conv = DoubleConv(in_channels, 16)
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)
        self.down4 = DownBlock(128, 256)

        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)

    def forward(self, x):
        x_input = self.input_conv(x)
        x1 = self.down1(x_input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.down4(x3)
        x = self.up1(x, x3)
        return x
        