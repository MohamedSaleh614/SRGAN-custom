import torch
import torch.nn as nn

class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels * 5, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)

        out2 = torch.cat([out1, x], dim=1)
        out2 = self.conv2(out2)
        out2 = self.relu(out2)

        out3 = torch.cat([out1, out2, x], dim=1)
        out3 = self.conv3(out3)
        out3 = self.relu(out3)

        out4 = torch.cat([out1, out2, out3, x], dim=1)
        out4 = self.conv4(out4)
        out4 = self.relu(out4)

        out5 = torch.cat([out1, out2, out3, out4, x], dim=1)
        out5 = self.conv5(out5)
        return out5 * 0.2 + x

class Generator(nn.Module):
    def __init__(self, in_channels, num_blocks=6):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[RRDB(64) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.conv2(out)
        return out + x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.conv5(out)
        return out
