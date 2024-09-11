import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Generator, self).__init__()
        self.project_layer = nn.Linear(input_channel, 4*4*1024)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        for param in self.parameters():
            nn.init.normal_(param, mean=0., std=0.02)
    
    def forward(self, z):
        # shape of z: (bs, 100)
        batch_size, input_channel = z.shape
        x = self.project_layer(z).reshape(batch_size, 1024, 4, 4)
        x = self.net(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channel):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1, kernel_size=4),
            nn.Sigmoid()
        )
        for param in self.parameters():
            nn.init.normal_(param, mean=0., std=0.02)
    
    def forward(self, x):
        # shape of x: (bs, 3, h, w)
        out = self.net(x).squeeze()
        return out
