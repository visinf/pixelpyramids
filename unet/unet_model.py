""" Full assembly of the parts to form the complete network """
import sys
import torch.nn.functional as F

from .unet_parts import *

'''class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, isfine=True, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.is_fine = isfine
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
        self.condup = CondUp()
        self.outc = OutConv(64, n_classes)

    def forward(self, x=None):
        if self.is_fine:
            x = self.condup(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits'''

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers, isfine=True, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.is_fine = isfine
        self.bilinear = bilinear
        self.n_layers = n_layers
        print(" ------------- " + str(n_layers) + "-----------------")
        print('downs ----')
        out_channels = 64
        self.inc = DoubleConv(n_channels, out_channels)
        self.downs = []
        for l_idx in range(0,n_layers-1):
            self.downs.append(Down(out_channels*(2**l_idx), out_channels*(2**(l_idx+1))))
            print(out_channels*(2**l_idx), out_channels*(2**(l_idx+1)))

        self.downs.append(Down(out_channels*(2**(n_layers-1)), out_channels*(2**(n_layers-1))))
        print(out_channels*(2**(n_layers-1)), out_channels*(2**(n_layers-1)))
        print('ups ----')
        self.ups = []
        for l_idx in reversed(range(1,n_layers)):
            #print('l_idx ',l_idx)
            self.ups.append(Up(out_channels*(2**(l_idx+1)), out_channels*(2**(l_idx-1)), bilinear))
            print(out_channels*(2**(l_idx+1)), out_channels*(2**(l_idx-1)))

        #, self_attn = (l_idx < 2 and n_layers >= 4)
        self.ups.append(Up(out_channels*(2**1), out_channels*(2**0), bilinear))
        print(out_channels*(2**1), out_channels*(2**0))
        print("-------------------------------------------------------")

        self.downs = nn.ModuleList(self.downs)
        self.ups = nn.ModuleList(self.ups)
        self.condup = CondUp()
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x=None,y=None):
        if self.is_fine:
            x = self.condup(x)

        x1 = self.inc(x)

        if self.n_layers > 1:

            down_outs = [x1]
            #print('x1 before ',x1.size())
            for layer in self.downs:
                x1 = layer(x1)
                #print('x1 ',x1.size())
                down_outs.append(x1)

            #print('All down_outs ',[x.size() for x in down_outs])
            x_n = self.ups[0](down_outs[-1], down_outs[-2])
            #print('x_n ',x_n.size())

            for i, layer in enumerate(self.ups[1:]):
                x_n = layer(x_n, down_outs[-(i+3)])
                #print('x_n ',x_n.size())
        else:
            x_n = x1

        logits = self.outc(x_n)
        return logits
