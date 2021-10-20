from __future__ import print_function
import argparse
import os
import numpy as np
from itertools import chain
from tqdm import tqdm
import json

import torch
from unet.unet_lstm import UNet as UNetLSTM
from unet.pixelcnnpp import PixelCNNpp
from utils import get_dataset

from misc import discretized_mix_logistic_loss, generate_samples
from optim import Adam
import torch.optim.lr_scheduler as sched

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--params_file', default='./params/lsun_bedroom.json', help='Configuration filename.')
	parser.add_argument('--gen_samples', action='store_true', help='Evaluate model.')
	parser.add_argument('--sampling_batch_size', type=int, default=4, help='Batch size for sampling')
	parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate.')
	
	args = parser.parse_args()
	with open(args.params_file) as json_data:
		config = json.load(json_data)["params"]

	dataset_name = config["dataset_name"]
	data_root = config["data_root"]
	batch_size = int(config["batch_size"])
	L = int(config["L"])
	out_channels = int(config["C"])
	n_bits = int(config["n_bits"])
	eval_interval = int(config["eval_interval"])
	n_classes = config["n_classes"]
	n_squeeze = config["n_squeeze"]
	#parameters for pixelcnnpp
	n_channels = config["n_channels"]
	n_res_layers = config["n_res_layers"]
	
	setting_id = str(dataset_name) + '_' + str(batch_size) + '_' + str(n_bits)

	
	
	unets = []
	for i in range(L-1):#
		if L ==11:
			n_layers = min(max(L - i//2 - 6,1),4)
		else:
			n_layers = min(max(L - i - 7,1),4)
		unets.append(torch.nn.DataParallel(UNetLSTM(nc_input=3, nc_cond_embd=100, n_classes=n_classes[i],
		 n_layers=n_layers,n_squeeze=n_squeeze[i], out_channels = out_channels)).cuda())
	
	unets.append(torch.nn.DataParallel(PixelCNNpp(image_dims=(3,4,4), n_channels=n_channels, n_res_layers=n_res_layers, n_classes=n_classes[-1])).cuda())

	train_loader, test_loader, image_shape = get_dataset( dataset_name, batch_size, data_root, L, n_bits )
	params = [list(unet.parameters()) for unet in unets]
	
	optimizerG = Adam(chain(*params), lr=0.0005, betas=(0.95, 0.9995), polyak=0.9995)
	scheduler = sched.ExponentialLR(optimizerG, 0.999995)
	epochs = 100
	best_loss = 99

	if not args.gen_samples:
		for epoch in range(epochs):

			train_bar = tqdm(train_loader)
			for i, data in enumerate(train_bar):
				out = []
				loss = 0.
				optimizerG.zero_grad()
				for l in range(L-1):
					out.append(unets[l]((data['downsmp_im'][l]).cuda(),(data['diff_im'][l]).cuda(), train=True))
					loss = loss+discretized_mix_logistic_loss(out[l],(data['diff_im'][l]).cuda(),n_bits)

				out.append(unets[L-1](data['downsmp_im'][-1].cuda()) )
				loss = loss+discretized_mix_logistic_loss(out[L-1],(data['downsmp_im'][-1]).cuda(),n_bits)
				avg_loss = torch.mean(loss,dim=0)

				
				avg_loss.backward()
				optimizerG.step()


				scheduler.step()
				train_bar.set_description('Train NLL (bits/dim) %.2f | Epoch %d -- Iteration ' % (avg_loss.item()/(np.log(2) * np.prod(image_shape)),epoch))
				
			if epoch % eval_interval == 0:
				
				all_loss =[]
				all_loss_per_level = {}
				for l in range(L-1):
					all_loss_per_level[l] = 0

				for i, data in enumerate(test_loader):
					out = []
					with torch.no_grad():
						tot_loss = 0.
						for l in range(L-1):
							out.append(unets[l]((data['downsmp_im'][l]).cuda(),(data['diff_im'][l]).cuda(), train=True))
							curr_loss = discretized_mix_logistic_loss(out[l],(data['diff_im'][l]).cuda(),n_bits)
							tot_loss = tot_loss + curr_loss
							all_loss_per_level[l] += torch.mean(curr_loss, dim=0).item()
						out.append(unets[L-1](data['downsmp_im'][-1].cuda()))
						tot_loss= tot_loss+discretized_mix_logistic_loss(out[L-1],(data['downsmp_im'][-1]).cuda(),n_bits)
						all_loss.append(tot_loss.detach().cpu().numpy())

				all_loss = np.concatenate(all_loss, axis=0)
				all_loss = np.sum(all_loss)/(np.log(2) * np.prod(image_shape) * len(test_loader) * batch_size)

				if float(all_loss) < best_loss:
					save_unets = {}
					for j in range(L):
						save_unets[j] = unets[j].module.state_dict()
						
					save_unets['optim'] = optimizerG.state_dict()
					save_unets['sched'] = scheduler.state_dict()
					torch.save(save_unets, os.path.join('./ckpts/', setting_id + '.pt'))

				best_loss = min(best_loss,float(all_loss))

				tqdm.write('Best Test NLL (bits/dim) at Epoch %d -- %.3f \n' % (epoch,best_loss))
	else:
		state_dicts = torch.load(os.path.join('./ckpts/', setting_id + '.pt'))#_0222
		for i in range(L):
			unets[i].load_state_dict(state_dicts[i])

		save_path = os.path.join('./samples/',setting_id)
		if not os.path.exists(save_path):#'./samples_celeba_cherrypick/'
			os.makedirs(save_path)

		generate_samples(unets,save_path,max_levels=L,n_bits=n_bits,
			num_samples=args.num_samples, batch_size=args.sampling_batch_size)
			

