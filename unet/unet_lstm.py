""" Full assembly of the parts to form the complete network """
import sys
import torch.nn.functional as F
import torchvision.utils as vutils

from .unet_parts import *
from convlstm.convlstm_model import *


def squeeze2d(input, factor=2):
	#assert factor >= 1 and isinstance(factor, int)
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
	x = input.view(B, C, H // factor, factor, W // factor, factor)
	x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
	x = x.view(B, C * factor * factor, H // factor, W // factor)
	return x


def unsqueeze2d(input, factor=2):
	assert factor >= 1 and isinstance(factor, int)
	factor2 = factor ** 2
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert C % (factor2) == 0, "{}".format(C)
	x = input.view(B, C // factor2, factor, factor, H, W)
	x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
	x = x.view(B, C // (factor2), H * factor, W * factor)
	return x

class SqueezeLayer(nn.Module):
	def __init__(self, factor):
		super(SqueezeLayer, self).__init__()
		self.factor = factor

	def forward(self, input, reverse=False):
		if not reverse:
			output = squeeze2d(input, self.factor)
			return output
		else:
			output = unsqueeze2d(input, self.factor)
			return output


class UNet(nn.Module):
	def __init__(self, nc_input, nc_cond_embd, n_classes, n_layers, n_squeeze, out_channels):
		super(UNet, self).__init__()
		self.nc_input = nc_input
		self.nc_cond_embd = nc_cond_embd
		self.n_classes = n_classes
		self.n_squeeze = n_squeeze
		self.out_channels = out_channels
		self.inc = DoubleConv(nc_input, out_channels)
		self.downs = []
		for l_idx in range(0,n_layers-1):
			self.downs.append(Down(out_channels*(2**l_idx), out_channels*(2**(l_idx+1))))

		self.downs.append(Down(out_channels*(2**(n_layers-1)), out_channels*(2**(n_layers-1))))
		self.ups = []
		for l_idx in reversed(range(1,n_layers)):
			self.ups.append(Up(out_channels*(2**(l_idx+1)), out_channels*(2**(l_idx-1))))

		self.ups.append(Up(out_channels*(2**1), out_channels*(2**0)))

		self.downs = nn.ModuleList(self.downs)
		self.ups = nn.ModuleList(self.ups)
		self.outc = OutConv(out_channels, nc_cond_embd)
		self.squeeze2d = SqueezeLayer(factor=2**n_squeeze)
		self.conv_lstm_enc = ConvLSTMEnc(input_ch=3,cond_ch=nc_cond_embd,out_ch=n_classes,hidden_size=128)

	def channel_squeeze(self, y):
		squeeze_diff_chs = []

		for ch_idx in range(y.size(1)):
			diff_sq = y[:,ch_idx:ch_idx+1,:,:]
			diff_sq = self.squeeze2d(diff_sq)
			squeeze_diff_chs.append(diff_sq)

		for ch_idx in range(len(squeeze_diff_chs)):
			squeeze_diff_chs[ch_idx] = squeeze_diff_chs[ch_idx].unsqueeze(2)


		return torch.cat(squeeze_diff_chs, dim=2)

	def channel_unsqueeze(self, y):
		squeeze_diff_chs = []
		for ch_idx in range(y.size(2)):
			diff_sq = y[:,:,ch_idx,:,:]
			diff_sq = self.squeeze2d(diff_sq, reverse=True)
			squeeze_diff_chs.append(diff_sq)
		return torch.cat(squeeze_diff_chs, dim=1)


	def forward(self, x=None,y=None,train=True):

		x1 = self.inc(x)

		down_outs = [x1]
		for layer in self.downs:
			x1 = layer(x1)
			down_outs.append(x1)

		x_n = self.ups[0](down_outs[-1], down_outs[-2])

		for i, layer in enumerate(self.ups[1:]):
			x_n = layer(x_n, down_outs[-(i+3)])

		enc_cond = self.outc(x_n)

		if self.n_squeeze > 0:
			squeeze_out = self.channel_squeeze(enc_cond)           
			if y is not None:
				squeeze_diff = self.channel_squeeze(y)
			else:
				squeeze_diff = None

			logits = self.conv_lstm_enc(squeeze_diff,squeeze_out,train=train)
			logits = self.channel_unsqueeze(logits)

		else:
			logits = enc_cond
		
		return logits
