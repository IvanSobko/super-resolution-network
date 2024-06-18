import torch
import torch.nn as nn

class BasicSRModel(nn.Module):

    def __init__(self, input_size=3, output_size=3, kernel=3, n_channels=64, block_num=10, upscale_factor=2):
        super(BasicSRModel, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        layers = []
        layers.append(nn.Conv2d(in_channels=input_size, out_channels=n_channels, kernel_size=kernel, padding=1))
        for _ in range(block_num):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=output_size, kernel_size=kernel, padding=1))
        
        layers = nn.Sequential(*layers)
        self.layers = layers


    def forward(self, x):
        output = self.upsample(x)
        return self.layers(output)
    
class ResidualSRModel(nn.Module):

    def __init__(self, input_size=3, output_size=3, kernel=3, n_channels=64, block_num=10, upscale_factor=2):
        super(ResidualSRModel, self).__init__()

        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        layers = []
        layers.append(nn.Conv2d(in_channels=input_size, out_channels=n_channels, kernel_size=kernel, padding=1))
        for _ in range(block_num):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=output_size, kernel_size=kernel, padding=1))
        
        layers = nn.Sequential(*layers)
        self.layers = layers


    def forward(self, x):
        output = self.upsample(x)
        return output + self.layers(output)

