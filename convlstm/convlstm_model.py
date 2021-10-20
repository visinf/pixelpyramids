import numpy as np
import torch
import torch.nn as nn

from convlstm.lstm import ConvSeqEncoder
from misc import sample_from_discretized_mix_logistic


class ConvLSTMEnc(nn.Module):
	#nc: number of channels in x_1
	#

	def __init__(self,input_ch,cond_ch,out_ch,hidden_size=128, num_layers=2, kernel_size=3, dilation=1, dp_rate=0.0):
		super().__init__()
		
		self.prior_lstm = ConvSeqEncoder( input_ch=input_ch+cond_ch, out_ch=out_ch,
			kernel_size=kernel_size, dilation=dilation,
			embed_ch=hidden_size, num_layers=num_layers, dropout=dp_rate )

		self.dp_rate = dp_rate
		

	def get_likelihood(self,x,cond):	

		init_zero_input = torch.zeros(x.size(0),1,x.size(2),x.size(3),x.size(4))
		if x.is_cuda:
			init_zero_input = init_zero_input.cuda()
		x_in = torch.cat([init_zero_input,x[:,0:-1]],dim=1)

		seq_lengths = torch.LongTensor((np.ones((x.size(0),))*(x_in.size(1))).astype(np.int32))#.cuda()
		
		lstm_input = torch.cat([cond,x_in], dim=2)

		logits, _ = self.prior_lstm(lstm_input,seq_lengths)
		return logits

	def get_sample(self,cond):

		with torch.no_grad():
			init_zero_input = torch.zeros(cond.size(0),1,3,cond.size(3),cond.size(4))
			if cond.is_cuda:
				init_zero_input = init_zero_input.cuda()
			seq_lengths = torch.LongTensor((np.ones((cond.size(0),))).astype(np.int32))#.cuda()
			hidden = None
			z_out = []
			curr_ch = None
			for t_step in range(cond.size(1)):

				if curr_ch is not None:
					curr_ch = curr_ch.unsqueeze(1)
					lstm_input = torch.cat([cond[:,t_step:t_step+1,:,:,:],curr_ch], dim=2)
				else:
					lstm_input = torch.cat([cond[:,t_step:t_step+1,:,:,:],init_zero_input], dim=2)

				logits, hidden = self.prior_lstm(lstm_input,seq_lengths,hidden)
				curr_ch = sample_from_discretized_mix_logistic(logits[:,0,:,:,:], [3,])
				z_out.append(curr_ch[:,None,:,:])

			z_sample = torch.cat(z_out, dim=1)	
			return z_sample


	def forward(self, x, cond, train=True):	
		if train:
			return self.get_likelihood(x, cond)
		else:
			return self.get_sample(cond)