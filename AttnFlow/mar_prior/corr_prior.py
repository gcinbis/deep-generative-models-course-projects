import numpy as np
import torch
import torch.nn as nn

from mar_prior.lstm import ConvSeqEncoder

class ChannelPriorUniScale(nn.Module):
	def __init__(self,batch_size,nc,height,width,level,tot_levels,hidden_size=32, num_layers=1, dp_rate=0.2, plot=False):
		super().__init__()
		self.batch_size = batch_size
		self.height = height//(2**(level))
		self.width = width//(2**(level))
		if level != tot_levels:
			self.nc = nc * 2**(level)
		else:
			self.nc = nc * 2**(level+1)

		#print('self.height ', self.height, ' self.width ', self.width, ' self.nc ', self.nc)
		self.z1_cond_network = nn.Sequential(nn.Conv2d(self.nc, 32, 5, stride=1, padding=2),
			nn.ReLU(),
			nn.Conv2d(32, 4, 5, stride=1, padding=2)
			)
		kernel_sizes = [5,5,3,3,3,3,3]
		dilations = [2,1,1,1,1,1,1]
		if level != tot_levels:
			self.prior_lstm = ConvSeqEncoder( input_ch=5, out_ch=2,
				kernel_size=kernel_sizes[level-1], dilation=dilations[level-1],
				embed_ch=hidden_size, num_layers=num_layers, dropout=dp_rate )
		else:
			self.prior_lstm = ConvSeqEncoder( input_ch=1, out_ch=2,
				kernel_size=kernel_sizes[level-1], dilation=dilations[level-1],
				embed_ch=hidden_size, num_layers=num_layers, dropout=dp_rate )

		self.Log2PI = float(np.log(2 * np.pi))
		#self.prior_final_fc = nn.Linear(hidden_size,2*self.nc*self.height)
		self.dp_rate = dp_rate
		self.level = level
		self.tot_levels = tot_levels
		self.cond_dropout = nn.Dropout2d(dp_rate)
		self.plot = plot
		if plot:
			self.plot_cntr = 0
			self.plot_interval = 20
		#

	def plotter(self, means, logs, num_channels=6):
		pass

	def dropout_in(self,z2):
		prob = torch.rand(z2.size(0),z2.size(1))
		z2[prob < self.dp_rate] = 0
		z2_sums = torch.sum(z2, dim=(2,3,4))
		return z2
		
	def likelihood(self, mean, logs, z):
		return -0.5 * (logs.to(z.device) * 2. + ((z - mean.to(z.device)) ** 2) / torch.exp(logs.to(z.device) * 2.) + self.Log2PI)

	def get_likelihood(self,z):	

		if isinstance(z, tuple):
			z1, z2 = z
			z1_embd = self.z1_cond_network(z1.type(torch.FloatTensor).cuda())
			z1_embd = z1_embd.unsqueeze(1).repeat(1,z2.size(1),1,1,1)
			z2 = z2.unsqueeze(2)
		else:
			z1 = None	
			z2 = z
			z2 = z2.unsqueeze(2)

		in_ch_length = z2.size(1)
		out_ch_length = z2.size(1)
		#print('z ',z.size())

		if self.level != self.tot_levels:
			init_zero_input = torch.zeros(self.batch_size,1,1,self.height,self.width)
		else:
			init_zero_input = torch.zeros(self.batch_size,1,1,self.height,self.width)

		if z2.is_cuda:
			init_zero_input = init_zero_input.cuda()

		seq_lengths = torch.LongTensor((np.ones((self.batch_size,))*(in_ch_length)).astype(np.int32))#.cuda()
		
		z2_dp = self.dropout_in(z2.clone().to(z2.device))
		lstm_input = torch.cat([init_zero_input,z2_dp[:,0:-1,:]], dim=1)

		if self.level != self.tot_levels:
			lstm_input = torch.cat([lstm_input,z1_embd.to(lstm_input.device)], dim=2)

		z_mean_logs, _ = self.prior_lstm(lstm_input,seq_lengths)
		means = z_mean_logs[:,-out_ch_length:,0:1]
		logs = z_mean_logs[:,-out_ch_length:,1:2]
		log_likelihood = self.likelihood( means, logs, z2[:,-out_ch_length:] )
		return torch.sum(log_likelihood, dim=(1,2,3,4))

	def reparametrize(self, mean, logs):
		if mean.is_cuda:	
			z = (torch.randn(mean.size()).cuda())*torch.exp(logs) + mean#
		else:
			z = torch.randn(mean.size())*torch.exp(logs) + mean#
		return z

	def get_sample(self,z1=None):	
		with torch.no_grad():
			hidden = None

			if z1 is not None:

				z1_embd = self.z1_cond_network(z1)
				z1_embd = z1_embd.unsqueeze(1)

				init_zero_input = torch.zeros(self.batch_size,1,1,self.height,self.width)
				if torch.cuda.is_available():
					init_zero_input = init_zero_input.cuda()
				lstm_input = torch.cat([init_zero_input,z1_embd], dim=2) 

			else:
				lstm_input = torch.zeros(self.batch_size,1,1,self.height,self.width)

			seq_lengths = torch.LongTensor((np.ones((self.batch_size,))).astype(np.int32))

			if torch.cuda.is_available():
				lstm_input = lstm_input.cuda()
			
			z_out = []
			for _ in range(self.nc):
				z_mean_logs, hidden = self.prior_lstm(lstm_input,seq_lengths,hidden)
				mean = z_mean_logs[:,:,0:1]
				logs = z_mean_logs[:,:,1:2]
				z_sample = self.reparametrize(mean, logs)
				z_out.append(z_sample)
				lstm_input = z_sample

				if z1 is not None:
					lstm_input = torch.cat([lstm_input,z1_embd], dim=2)

			z_sample = torch.cat(z_out, dim=1)	
			z_sample = z_sample.squeeze(2)
			return z_sample


	def forward(self,z,reverse=False):	
		if not reverse:
			return self.get_likelihood(z)
		else:
			if z is not None:
				if isinstance(z, tuple):
					z1, _ = z
				else:
					z1 = z

				return self.get_sample(z1)	
			else:
				return self.get_sample()	


class ChannelPriorMultiScale(nn.Module):
	def __init__(self,batch_size,nc,height,width,levels,hidden_size=32,dp_rate=0., num_layers=2, mog=False):
		super().__init__()

		if not mog:
			self.prior_list = [ChannelPriorUniScale(batch_size,nc,height,width,level,levels,hidden_size=hidden_size,
				num_layers=num_layers,dp_rate=dp_rate, plot = (True if level == 2 else False)) 
			for level in range(1,levels+1)]
		else:
			raise NotImplementedError
		self.prior_list = nn.ModuleList(self.prior_list)

	def get_likelihood(self,z,level,hidden=None):	
		likelihood = self.prior_list[level-1](z,reverse=False)		
		return likelihood	

	def get_sample(self,z,level,hidden=None):
		z = self.prior_list[level-1](z,reverse=True)	
		return z

	def forward(self,z,level,reverse=False):	
		if not reverse:
			return self.get_likelihood(z,level)
		else:
			return self.get_sample(z,level)	
