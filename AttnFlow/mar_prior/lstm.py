import torch
import torch.nn as nn

from mar_prior.convolutional_rnn import Conv2dLSTM


class ConvSeqEncoder(nn.Module):
	def __init__(self, input_ch, out_ch, embed_ch, kernel_size=5, dilation=1, num_layers=1, bidirectional=False, dropout=0.0):
		super().__init__()
		self.lstm = Conv2dLSTM(in_channels=embed_ch,  # Corresponds to input size
								   out_channels=embed_ch,  # Corresponds to hidden size
								   kernel_size=kernel_size,  # Int or List[int]
								   num_layers=num_layers,
								   bidirectional=bidirectional,
								   dilation=dilation, stride=1, dropout=0.0,
								   batch_first=True)

		
		self.conv_embed = nn.Conv2d(input_ch, embed_ch, kernel_size, stride=1, padding=(1 if kernel_size==3 else 2))
		self.conv_out1 = nn.Conv2d(embed_ch * (2 if bidirectional else 1), out_ch, 3, stride=1, padding=1 )
		self.embed_ch = embed_ch
		self.out_ch = out_ch
		self.dropout = dropout
		self.conv_dropout = nn.Dropout2d(dropout)

	def td_conv(self,x,conv_fn,out_ch):
		x = x.contiguous()
		batch_size = x.size(0)
		time_steps = x.size(1)
		x = x.view(batch_size*time_steps,x.size(2),x.size(3),x.size(4))
		x = conv_fn(x.type(torch.FloatTensor).cuda())
		if self.dropout > 0:
			x = self.conv_dropout(x)	
		x = x.view(batch_size,time_steps,out_ch,x.size(2),x.size(3))
		return x
		
	def forward(self, x, lengths, hidden = None):
		x2 = self.td_conv(x,self.conv_embed,self.embed_ch)
		
		outputs, hidden = self.lstm(x2, hidden)
			
		output = self.td_conv(outputs,self.conv_out1,self.out_ch)
		return output, hidden