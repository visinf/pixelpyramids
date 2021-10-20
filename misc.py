from __future__ import print_function
import random
import numpy as np
import os

import torch
import torch.nn.functional as F

from scipy import ndimage
from sklearn.ensemble import IsolationForest
from PIL import Image

from tqdm import tqdm

def cpd_sum(tensor, dim=None, keepdim=False):
	if dim is None:
		# sum up all dim
		return torch.sum(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.sum(dim=d, keepdim=True)
		if not keepdim:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor

def cpd_mean(tensor, dim=None, keepdims=False):
	if dim is None:
		return tensor.mean(tensor)
	else:
		if isinstance(dim, int):
			dim = [dim]
		dim = sorted(dim)
		for d in dim:
			tensor = tensor.mean(dim=d, keepdim=True)
		if not keepdims:
			for i, d in enumerate(dim):
				tensor.squeeze_(d-i)
		return tensor	

def concat_elu(x):
	""" like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
	# Pytorch ordering
	axis = len(x.size()) - 3
	return F.elu(torch.cat([x, -x], dim=axis))


def log_sum_exp(x):
	""" numerically stable log_sum_exp implementation that prevents overflow """
	# TF ordering
	axis  = len(x.size()) - 1
	m, _  = torch.max(x, dim=axis)
	m2, _ = torch.max(x, dim=axis, keepdim=True)
	return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
	""" numerically stable log_softmax implementation that prevents overflow """
	# TF ordering
	axis = len(x.size()) - 1
	m, _ = torch.max(x, dim=axis, keepdim=True)
	return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(l, x, n_bits):
	""" log likelihood for mixture of discretized logistics
	Args
		l -- model output tensor of shape (B, 10*n_mix, H, W), where for each n_mix there are
				3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
		x -- data tensor of shape (B, C, H, W) with values in model space [-1, 1]
	"""
	# shapes
	B, C, H, W = x.shape
	n_mix = l.shape[1] // (1 + 3*C)

	# unpack params of mixture of logistics
	logits = l[:, :n_mix, :, :]                         # (B, n_mix, H, W)
	l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
	means, logscales, coeffs = l.split(n_mix, 1)        # (B, n_mix, C, H, W)
	logscales = logscales.clamp(min=-7)
	coeffs = coeffs.tanh()

	# adjust means of channels based on preceding subpixel (cf PixelCNN++ eq 3)
	x  = x.unsqueeze(1).expand_as(means)
	if C!=1:
		m1 = means[:, :, 0, :, :]
		m2 = means[:, :, 1, :, :] + coeffs[:, :, 0, :, :] * x[:, :, 0, :, :]
		m3 = means[:, :, 2, :, :] + coeffs[:, :, 1, :, :] * x[:, :, 0, :, :] + coeffs[:, :, 2, :, :] * x[:, :, 1, :, :]
		means = torch.stack([m1, m2, m3], 2)  # out (B, n_mix, C, H, W)

	# log prob components
	scales = torch.exp(-logscales)
	plus = scales * (x - means + 1/(2**n_bits-1))
	minus = scales * (x - means - 1/(2**n_bits-1))

	# partition the logistic pdf and cdf for x in [<-0.999, mid, >0.999]
	# 1. x<-0.999 ie edge case of 0 before scaling
	cdf_minus = torch.sigmoid(minus)
	log_one_minus_cdf_minus = - F.softplus(minus)
	# 2. x>0.999 ie edge case of 255 before scaling
	cdf_plus = torch.sigmoid(plus)
	log_cdf_plus = plus - F.softplus(plus)
	# 3. x in [-.999, .999] is log(cdf_plus - cdf_minus)

	# compute log probs:
	# 1. for x < -0.999, return log_cdf_plus
	# 2. for x > 0.999,  return log_one_minus_cdf_minus
	# 3. x otherwise,    return cdf_plus - cdf_minus
	log_probs = torch.where(x < (-1.00 + 2/(2**n_bits-1)), log_cdf_plus,
							torch.where(x > (1.00 - 2/(2**n_bits-1)), log_one_minus_cdf_minus,
										torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))))
	log_probs = log_probs.sum(2) + F.log_softmax(logits, 1) # log_probs sum over channels (cf eq 3), softmax over n_mix components (cf eq 1)

	# marginalize over n_mix components and return negative log likelihood per data point
	return - log_probs.logsumexp(1).sum([1,2])  # out (B,)


def discretized_mix_logistic_loss_1d(x, l):
	""" log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
	# Pytorch ordering
	x = x.permute(0, 2, 3, 1)
	l = l.permute(0, 2, 3, 1)
	xs = [int(y) for y in x.size()]
	ls = [int(y) for y in l.size()]

	# here and below: unpacking the params of the mixture of logistics
	nr_mix = int(ls[-1] / 3)
	logit_probs = l[:, :, :, :nr_mix]
	l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
	means = l[:, :, :, :, :nr_mix]
	log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
	# here and below: getting the means and adjusting them based on preceding
	# sub-pixels
	x = x.contiguous()
	x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

	# means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
	centered_x = x - means
	inv_stdv = torch.exp(-log_scales)
	plus_in = inv_stdv * (centered_x + 1. / 255.)
	cdf_plus = F.sigmoid(plus_in)
	min_in = inv_stdv * (centered_x - 1. / 255.)
	cdf_min = F.sigmoid(min_in)
	# log probability for edge case of 0 (before scaling)
	log_cdf_plus = plus_in - F.softplus(plus_in)
	# log probability for edge case of 255 (before scaling)
	log_one_minus_cdf_min = -F.softplus(min_in)
	cdf_delta = cdf_plus - cdf_min  # probability for all other cases
	mid_in = inv_stdv * centered_x
	# log probability in the center of the bin, to be used in extreme cases
	# (not actually used in our code)
	log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
	
	inner_inner_cond = (cdf_delta > 1e-5).float()
	inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
	inner_cond       = (x > 0.999).float()
	inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
	cond             = (x < -0.999).float()
	log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
	log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
	
	return -torch.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
	# we perform one hot encore with respect to the last axis
	one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
	if tensor.is_cuda : one_hot = one_hot.cuda()
	one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
	return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
	# Pytorch ordering
	l = l.permute(0, 2, 3, 1)
	ls = [int(y) for y in l.size()]
	xs = ls[:-1] + [1] #[3]

	# unpack parameters
	logit_probs = l[:, :, :, :nr_mix]
	l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # for mean, scale

	# sample mixture indicator from softmax
	temp = torch.FloatTensor(logit_probs.size())
	if l.is_cuda : temp = temp.cuda()
	temp.uniform_(1e-5, 1. - 1e-5)
	temp = logit_probs.data - torch.log(- torch.log(temp))
	_, argmax = temp.max(dim=3)
   
	one_hot = to_one_hot(argmax, nr_mix)
	sel = one_hot.view(xs[:-1] + [1, nr_mix])
	# select logistic parameters
	means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4) 
	log_scales = torch.clamp(torch.sum(
		l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
	u = torch.FloatTensor(means.size())
	if l.is_cuda : u = u.cuda()
	u.uniform_(1e-5, 1. - 1e-5)
	u = Variable(u)
	x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
	x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
	out = x0.unsqueeze(1)
	return out

def sample_from_discretized_mix_logistic(l, image_dims):
	# shapes
	B, _, H, W = l.shape
	C = image_dims[0]#3
	n_mix = l.shape[1] // (1 + 3*C)

	# unpack params of mixture of logistics
	logits = l[:, :n_mix, :, :]
	l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
	means, logscales, coeffs = l.split(n_mix, 1)  # each out (B, n_mix, C, H, W)
	logscales = logscales.clamp(min=-7)
	coeffs = coeffs.tanh()

	# sample mixture indicator
	argmax = torch.argmax(logits - torch.log(-torch.log(torch.rand_like(logits).uniform_(1e-5, 1 - 1e-5))), dim=1)
	sel = torch.eye(n_mix, device=logits.device)[argmax]#, dtype=torch.float32
	sel = sel.permute(0,3,1,2).unsqueeze(2)  # (B, n_mix, 1, H, W)

	# select mixture components
	means = means.mul(sel).sum(1)
	logscales = logscales.mul(sel).sum(1)
	coeffs = coeffs.mul(sel).sum(1)

	# sample from logistic using inverse transform sampling
	u = torch.rand_like(means).uniform_(1e-5, 1 - 1e-5)
	#u = torch.rand_like(means).uniform_(0.1, 1 - 0.1)#float(np.e ** (-1/0.7))  0.1*
	x = means + 0.1*logscales.exp() * (torch.log(u) - torch.log1p(-u))  # logits = inverse logistic 0.5 *  0.1*
	#
	if C==1:
		return x.clamp(-1,1)
	else:
		x0 = torch.clamp(x[:,0,:,:], -1, 1)
		x1 = torch.clamp(x[:,1,:,:] + coeffs[:,0,:,:] * x0, -1, 1)
		x2 = torch.clamp(x[:,2,:,:] + coeffs[:,1,:,:] * x0 + coeffs[:,2,:,:] * x1, -1, 1)
		return torch.stack([x0, x1, x2], 1)  # out (B, C, H, W)


def sample_coarse_pcnn(model, image_dims, batch_size):
    samps = torch.zeros(batch_size, *image_dims).cuda()
    for yi in range(image_dims[1]):
        for xi in range(image_dims[2]):
            l = model(samps, None)
            samps[:,:,yi,xi] = sample_from_discretized_mix_logistic(l, image_dims)[:,:,yi,xi]
    return samps

def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv
    return hsv

def batch_med_filter(im_batch, size=2):
	filtered_batch = []
	for im in im_batch:
		filtered_batch.append(ndimage.median_filter(im, size=size))
	return np.array(filtered_batch)

def get_grad_mask( img, l, n_bits=5 ):
	factor = (2**n_bits)//(2**5)
	img = np.mean(img, axis=1)
	dx_p = img - np.pad(img[:,1:,:],((0,0),(0,1),(0,0)),mode='edge') 
	dx_m = img - np.pad(img[:,:-1,:],((0,0),(1,0),(0,0)),mode='edge') 
	dy_p = img - np.pad(img[:,:,1:],((0,0),(0,0),(0,1)),mode='edge') 
	dy_m = img - np.pad(img[:,:,:-1],((0,0),(0,0),(1,0)),mode='edge') 
	d = (dx_p + dx_m + dy_p + dy_m)/4.
	mask = d > ((8.0 + l*0.25)*factor)
	return np.repeat(mask[:,None,:,:],3,axis=1)

def get_pixel_mask( img, n_bits=5 ):
	factor = (2**n_bits)//(2**5)
	r_m = np.logical_and.reduce(((img[:,0,:,:] - img[:,1,:,:]) > 18*factor, (img[:,0,:,:] - img[:,2,:,:]) > 18*factor))
	g_m = np.logical_and.reduce(((img[:,1,:,:] - img[:,0,:,:]) > 1*factor, (img[:,1,:,:] - img[:,2,:,:]) > 1*factor))
	b_m = np.logical_and.reduce(((img[:,2,:,:] - img[:,0,:,:]) > 4*factor, (img[:,2,:,:] - img[:,1,:,:]) > 4*factor))
	m_m = np.logical_and.reduce(((img[:,0,:,:] - img[:,1,:,:]) > 1*factor, (img[:,2,:,:] - img[:,1,:,:]) > 1*factor,
		np.abs(img[:,0,:,:] - img[:,2,:,:]) < 14*factor))
	c_m = np.logical_and.reduce(((img[:,1,:,:] - img[:,0,:,:]) > 1*factor, (img[:,2,:,:] - img[:,0,:,:]) > 1*factor,
		np.abs(img[:,1,:,:] - img[:,2,:,:]) < 6*factor))
	y_m = np.logical_and.reduce(((img[:,0,:,:] - img[:,2,:,:]) > 1*factor, (img[:,1,:,:] - img[:,2,:,:]) > 1*factor,
		np.abs(img[:,1,:,:] - img[:,0,:,:]) < 1*factor))
	mask = np.logical_or.reduce((r_m,g_m,b_m,m_m,c_m,y_m))
	return np.repeat(mask[:,None,:,:],3,axis=1)

def clean_coarse_level( coarse_samps, n_bits=5 ):
	coarse_samps_med = batch_med_filter(coarse_samps)
	pixel_mask = get_pixel_mask( coarse_samps, n_bits )
	coarse_samps[pixel_mask] = coarse_samps_med[pixel_mask]
	return coarse_samps

def get_final_outliers( im_batch ):
	mask = []
	for img in im_batch: 
		outlier_det_data = [np.zeros((1*12,)) for _ in range(img.shape[1]*img.shape[2])]
		img_hsv = rgb2hsv(np.transpose(img, (1,2,0)))
		for i in range(img.shape[1]):
			for j in range(img.shape[2]):
				if i > 32 and i < img.shape[1] - 32 and j > 32 and j < img.shape[2] - 32:
					neighbour_hist, _ = np.histogram(img_hsv[i+3 - 2:i+3 + 3,j+3 - 2:j+3 + 3,0].flatten(),bins=12,range=(0,360))
					outlier_det_data[i * img.shape[1] + j] = neighbour_hist.flatten()

		outlier_det_data = np.array(outlier_det_data)	
		cov = IsolationForest(n_estimators=100,random_state=0,contamination=0.05).fit(outlier_det_data)#support_fraction=0.95,
		outliers = cov.predict(outlier_det_data) < 0
		outliers = outliers.reshape((img.shape[1],img.shape[2]))
		mask.append(outliers)
	mask = np.array(mask)
	return np.repeat(mask[:,None,:,:],3,axis=1)

def get_next_image(downsmp_im,diff_sample, n_bits, l, postprocess=True):
	n_bins = 2**n_bits
	if downsmp_im.shape[-2] != downsmp_im.shape[-1]:
		up_sample = np.zeros((downsmp_im.shape[0],downsmp_im.shape[1],2*downsmp_im.shape[2],downsmp_im.shape[3]))#.cuda()
		diff_sample = np.mod( diff_sample.astype(np.uint8) - 128, n_bins)
		up_sample_even_rows = np.mod(downsmp_im.astype(np.uint8) + diff_sample.astype(np.uint8), n_bins)
		up_sample_even_rows = up_sample_even_rows.astype(np.double)
		up_sample[:,:,0::2,:] = up_sample_even_rows
		up_sample[:,:,1::2,:] = downsmp_im
	else:
		up_sample = np.zeros((downsmp_im.shape[0],downsmp_im.shape[1],downsmp_im.shape[2],2*downsmp_im.shape[3]))
		diff_sample = np.mod( diff_sample.astype(np.uint8) - 128, n_bins)
		up_sample_even_rows = np.mod(downsmp_im.astype(np.uint8) + diff_sample.astype(np.uint8), n_bins)
		up_sample_even_rows = up_sample_even_rows.astype(np.double)
		up_sample[:,:,:,0::2] = up_sample_even_rows
		up_sample[:,:,:,1::2] = downsmp_im

	pixel_mask = get_pixel_mask( up_sample, n_bits )
	grad_mask = get_grad_mask(up_sample, l, n_bits)
	full_mask = np.logical_or.reduce((grad_mask,pixel_mask))
	up_sample_med = batch_med_filter(up_sample,size=3)
	up_sample[full_mask] = up_sample_med[full_mask]

	if postprocess and l == 0:
		final_mask = get_final_outliers(up_sample)
		up_sample_med = batch_med_filter(up_sample,size=8)
		up_sample[final_mask] = up_sample_med[final_mask]
	
	return up_sample


def generate_samples(unets,save_path,max_levels,n_bits,num_samples=32,batch_size=4):
	all_samps = []
	coarsest_image_dims  = [3,4,4]

	for _ in tqdm(range(num_samples//batch_size)):
		up_sample = sample_coarse_pcnn(unets[max_levels-1], coarsest_image_dims, batch_size)
		up_sample = clean_coarse_level(up_sample.detach().cpu().numpy())
		for l in reversed(range(max_levels-1)):
			curr_shape  = [3,np.power(2,max_levels-l-1),np.power(2,max_levels-l-1)]
			out = unets[l](x=torch.tensor(up_sample).float().cuda(),y=None,train=False)
			diff_sample = out.detach().cpu().numpy().astype(np.double)
			diff_sample = (diff_sample + 1.)/2.
			diff_sample = (diff_sample*((2**n_bits) - 1))

			up_sample = (up_sample + 1.)/2.
			up_sample = (up_sample*((2**n_bits) - 1))

			up_sample = get_next_image(up_sample,diff_sample, n_bits, l)

			up_sample = up_sample / ((2**n_bits) - 1)
			up_sample = 2.*up_sample
			up_sample = up_sample - 1.

		up_sample = (up_sample + 1.)/2.

		all_samps.append(up_sample)

	all_samps = np.concatenate(all_samps, axis=0)
	for idx, im_final in enumerate(all_samps):
		im_final = np.transpose((im_final*255).astype(np.uint8),(1,2,0))
		im_final = Image.fromarray(im_final)
		im_final.save(os.path.join(save_path, str(idx).rjust(4,'0') + '.png'))


