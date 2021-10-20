"""
PixelCNN++ implementation following https://github.com/openai/pixel-cnn/

References:
    1. Salimans, PixelCNN++ 2017
    2. van den Oord, Pixel Recurrent Neural Networks 2016a
    3. van den Oord, Conditional Image Generation with PixelCNN Decoders, 2016c
    4. Reed 2016 http://www.scottreed.info/files/iclr2017.pdf
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


# --------------------
# Helper functions
# --------------------

def down_shift(x):
#    B, C, H, W = x.shape
#    return torch.cat([torch.zeros([B, C, 1, W], device=x.device), x[:,:,:H-1,:]], 2)
    return F.pad(x, (0,0,1,0))[:,:,:-1,:]

def right_shift(x):
#    B, C, H, W = x.shape
#    return torch.cat([torch.zeros([B, C, H, 1], device=x.device), x[:,:,:,:W-1]], 3)
    return F.pad(x, (1,0))[:,:,:,:-1]

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

# --------------------
# Model components
# --------------------

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class DownShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad H above and W on each side
        Hk, Wk = self.kernel_size
        x = F.pad(x, ((Wk-1)//2, (Wk-1)//2, Hk-1, 0))
        return super().forward(x)

class DownRightShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad above and on left (ie shift input down and right)
        Hk, Wk = self.kernel_size
        x = F.pad(x, (Wk-1, 0, Hk-1, 0))
        return super().forward(x)

class DownShiftedConvTranspose2d(ConvTranspose2d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Hout, Wout = x.shape
        Hk, Wk = self.kernel_size
        Hs, Ws = self.stride
#        return x[:, :, :Hout - Hk + 1, (Wk-1)//2: Wout - (Wk-1)//2]
        return x[:, :, :Hout-Hk+Hs, (Wk)//2: Wout]  # see pytorch doc for ConvTranspose output

class DownRightShiftedConvTranspose2d(ConvTranspose2d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Hout, Wout = x.shape
        Hk, Wk = self.kernel_size
        Hs, Ws = self.stride
#        return x[:, :, :Hout+1-Hk, :Wout+1-Wk]  # see pytorch doc for ConvTranspose output
        return x[:, :, :Hout-Hk+Hs, :Wout-Wk+Ws]  # see pytorch doc for ConvTranspose output

class GatedResidualLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2*shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, a=None, h=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:  # shortcut connection if auxiliary input 'a' is given
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if h is not None:
            c2 += self.proj_h(h)[:,:,None,None]
        a, b = c2.chunk(2,1)
        out = x + a * torch.sigmoid(b)
        return out

class CondConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.c1 = Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()#concat_elu #
        '''self.c2 = Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, padding=padding)'''

        #self.proj_in = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x ):
        #residual = x
        #x = self.c2(self.relu(self.c1(x)))
        return self.relu(self.c1(x))#self.proj_in(residual) + x

class CondGatedResidualLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2*shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        self.proj_h = conv(n_channels, 2*n_channels, kernel_size)

    def forward(self, x, a=None, h=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:  # shortcut connection if auxiliary input 'a' is given
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if h is not None:
            #print('h size ',h.size())
            c3 = self.proj_h(h)
        a, b = c2.chunk(2,1)
        c, d = c3.chunk(2,1)
        out = x + a * torch.sigmoid(b) + c * torch.sigmoid(d)#c3#
        return out

# --------------------
# PixelCNN
# --------------------

class DeepPixelCNNpp(nn.Module):
    def __init__(self, image_dims=(3,4,4), n_channels=32, n_res_layers=2, n_classes=10, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])

        self.up_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)]])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        self.down_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        # output logistic mix params
        #   each component has 3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        self.output_conv = Conv2d(n_channels, n_classes, kernel_size=1)#(3*image_dims[0]+1)*n_logistic_mix

    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]
        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]

        # up pass
        for u_module, ul_module in zip(self.up_u_modules, self.up_ul_modules):
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, GatedResidualLayer) else u_module(u_list[-1])]
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, GatedResidualLayer) else [ul_module(ul_list[-1])]

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()
        for u_module, ul_module in zip(self.down_u_modules, self.down_ul_modules):
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, GatedResidualLayer) else u_module(u)
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, GatedResidualLayer) else ul_module(ul)

        return self.output_conv(F.elu(ul))

class VeryDeepPixelCNNpp(nn.Module):
    def __init__(self, image_dims=(3,4,4), n_channels=32, n_res_layers=2, n_classes=10, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])

        self.up_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)]])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        self.down_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        # output logistic mix params
        #   each component has 3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        self.output_conv = Conv2d(n_channels, n_classes, kernel_size=1)#(3*image_dims[0]+1)*n_logistic_mix

    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]
        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]

        # up pass
        for u_module, ul_module in zip(self.up_u_modules, self.up_ul_modules):
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, GatedResidualLayer) else u_module(u_list[-1])]
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, GatedResidualLayer) else [ul_module(ul_list[-1])]

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()
        for u_module, ul_module in zip(self.down_u_modules, self.down_ul_modules):
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, GatedResidualLayer) else u_module(u)
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, GatedResidualLayer) else ul_module(ul)

        return self.output_conv(F.elu(ul))


class PixelCNNpp(nn.Module):
    def __init__(self, image_dims=(3,4,4), n_channels=32, n_res_layers=2, n_classes=10, n_cond_classes=None, drop_rate=0.0):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            ])

        self.up_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            ])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            ])

        self.down_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            ])

        # output logistic mix params
        #   each component has 3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        self.output_conv = Conv2d(n_channels, n_classes, kernel_size=1)#(3*image_dims[0]+1)*n_logistic_mix

    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]
        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]

        # up pass
        for u_module, ul_module in zip(self.up_u_modules, self.up_ul_modules):
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, GatedResidualLayer) else u_module(u_list[-1])]
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, GatedResidualLayer) else [ul_module(ul_list[-1])]

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()
        for u_module, ul_module in zip(self.down_u_modules, self.down_ul_modules):
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, GatedResidualLayer) else u_module(u)
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, GatedResidualLayer) else ul_module(ul)

        return self.output_conv(F.elu(ul))


class CondPixelCNNpp(nn.Module):
    def __init__(self, image_dims=(3,4,4), n_channels=32, n_res_layers=2, n_classes=10, cond_dims=None, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))

        self.cond_input  = nn.Conv2d(cond_dims[0], n_channels, kernel_size=(3,3), padding=1)
        self.cond_mid  = nn.Conv2d(n_channels, n_channels, kernel_size=(3,3), padding=1)

        self.up_cond_modules = nn.ModuleList([
            *[CondConvLayer(n_channels, n_channels, (3,3), 1) for _ in range(n_res_layers)],
            nn.Conv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondConvLayer(n_channels, n_channels, (3,3), 1) for _ in range(n_res_layers)],
            nn.Conv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondConvLayer(n_channels, n_channels, (3,3), 1) for _ in range(n_res_layers)]])

        self.down_cond_modules = nn.ModuleList([
            *[CondConvLayer(2*n_channels, n_channels, (3,3), 1) for _ in range(n_res_layers)],
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondConvLayer(2*n_channels, n_channels, (3,3), 1) for _ in range(n_res_layers)],
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondConvLayer(2*n_channels, n_channels, (3,3), 1) for _ in range(n_res_layers)]])

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[CondGatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[CondGatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[CondGatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])

        self.up_ul_modules = nn.ModuleList([
            *[CondGatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondGatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondGatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)]])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[CondGatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[CondGatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[CondGatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        self.down_ul_modules = nn.ModuleList([
            *[CondGatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondGatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[CondGatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        # output logistic mix params
        #   each component has 3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        self.output_conv = Conv2d(n_channels, n_classes, kernel_size=1)#(3*image_dims[0]+1)*n_logistic_mix

    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]
        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]
        cond_list_up = [self.cond_input(h)]

        for cond_module in self.up_cond_modules:
            cond_list_up += [cond_module(cond_list_up[-1])]

        #print('cond_list_up ',[x.size() for x in cond_list_up])

        cond_list_down = [self.cond_mid(cond_list_up[-1])]
        

        for cond_module, h in zip(self.down_cond_modules,reversed(cond_list_up)):
            cond_list_down += [cond_module(torch.cat([cond_list_down[-1],h], dim=1) \
                if isinstance(cond_module, CondConvLayer) else cond_list_down[-1])]

        #print('cond_list_down ',[x.size() for x in cond_list_down])

        # up pass
        for u_module, ul_module, h  in zip(self.up_u_modules, self.up_ul_modules, cond_list_up):
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, CondGatedResidualLayer) else u_module(u_list[-1])]
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, CondGatedResidualLayer) else [ul_module(ul_list[-1])]
        
        #print('u_list ',[x.size() for x in u_list])
        #print('ul_list ',[x.size() for x in ul_list])
        #sys.exit(0)
        # down pass
        u = u_list.pop()
        ul = ul_list.pop()
        for u_module, ul_module, h in zip(self.down_u_modules, self.down_ul_modules, cond_list_down):
            #print('u, ul before ',u.size(),ul.size(), h.size())
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, CondGatedResidualLayer) else u_module(u)
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, CondGatedResidualLayer) else ul_module(ul)
            #print('u, ul before ',u.size(),ul.size())
        #sys.exit(0)
        return self.output_conv(F.elu(ul))
