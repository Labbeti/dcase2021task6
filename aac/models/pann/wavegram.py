"""
	Based on class "Wavegram_Logmel_Cnn14" from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
"""
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

from .models import ConvBlock, ConvPreWavBlock, init_bn, init_layer


class Wavegram(nn.Module):
	def __init__(self, fpath: Optional[str] = None, trainable: bool = True, reshape: bool = True, **kwargs) -> None:
		"""
			:param fpath: The optional path to the pre-trained weights of Wavegram model. (default: None)
			:param trainable: If True, the parameters can be modified by back-propagation (default: True)
			:param reshape: If True, reshape the output to a tensor of shape (bsize, embed_len, seq_len). (default: True)
		"""
		super().__init__()
		self.fpath = fpath
		self.trainable = trainable
		self.reshape = reshape

		self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
		self.pre_bn0 = nn.BatchNorm1d(64)
		self.pre_block1 = ConvPreWavBlock(64, 64)
		self.pre_block2 = ConvPreWavBlock(64, 128)
		self.pre_block3 = ConvPreWavBlock(128, 128)
		self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)
		
		if fpath is None:
			self.init_weight()
		else:
			if not osp.isfile(fpath):
				raise RuntimeError(f'Cannot use pre-trained weights for Wavegram (fpath "{fpath}" is not a file).')
			data = torch.load(fpath, map_location=torch.device('cpu'))
			self.load_state_dict(data['model'], strict=False)

		for param in self.parameters():
			param.requires_grad = trainable

	def init_weight(self) -> None:
		init_layer(self.pre_conv0)
		init_bn(self.pre_bn0)
 
	def forward(self, input_: Tensor) -> Tensor:
		"""
			Input: (batch_size, data_length) or (data_length,) tensor
		"""
		if len(input_.shape) not in [1, 2]:
			raise RuntimeError(
				f'Invalid number of dimensions for Wavegram input. Maybe use raw audios for compute the wavegram representation. '
				f'(number of dims is {len(input_.shape)}, but expects 1 or 2).'
			)

		squeeze_at_end = False
		if len(input_.shape) == 1:
			input_ = input_.unsqueeze(dim=0)
			squeeze_at_end = True

		# Wavegram
		a1 = F.relu_(self.pre_bn0(self.pre_conv0(input_[:, None, :])))
		a1 = self.pre_block1(a1, pool_size=4)
		a1 = self.pre_block2(a1, pool_size=4)
		a1 = self.pre_block3(a1, pool_size=4)
		a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
		a1 = self.pre_block4(a1, pool_size=(2, 1))

		if self.reshape:
			assert len(a1.shape) == 4
			# (bsize, A, seq, B) -> (bsize, A * B, seq)
			a1 = a1.transpose(2, 3)
			a1 = a1.contiguous()
			a1 = a1.reshape(a1.shape[0], -1, a1.shape[-1])
			assert len(a1.shape) == 3
		
		if squeeze_at_end:
			a1 = a1.squeeze(dim=0)

		return a1
