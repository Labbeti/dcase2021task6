"""
Provides functions to use SpecAugment
IMPORTED FROM https://github.com/qiuqiangkong/sound_event_detection_dcase2017_task4
"""

import torch
from torch import nn


class DropStripes(nn.Module):
	def __init__(self, dim, drop_width, stripes_num):
		"""Drop stripes.

		Args:
			dim: int, dimension along which to drop
			drop_width: int, maximum width of stripes to drop
			stripes_num: int, how many stripes to drop
		"""
		super(DropStripes, self).__init__()

		assert dim in [2, 3]  # dim 2: time; dim 3: frequency

		self.dim = dim
		self.drop_width = drop_width
		self.stripes_num = stripes_num

	def forward(self, input):
		"""input: (batch_size, channels, time_steps, freq_bins)"""

		assert input.ndimension() == 4

		if self.training is False:
			return input

		else:
			batch_size = input.shape[0]
			total_width = input.shape[self.dim]

			for n in range(batch_size):
				self.transform_slice(input[n], total_width)

			return input

	def transform_slice(self, e, total_width):
		"""e: (channels, time_steps, freq_bins)"""
		# print("stripes_num", self.stripes_num)
		for _ in range(self.stripes_num):
			distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
			bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

			if self.dim == 2:
				# print(i, "T", bgn, bgn+distance)
				e[:, bgn: bgn + distance, :] = 0
			elif self.dim == 3:
				# print(i, "F", bgn, bgn+distance)
				e[:, :, bgn: bgn + distance] = 0


class SpecAugmentation(nn.Module):
	def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num):
		"""Spec augmetation.
		[ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
		and Le, Q.V., 2019. Specaugment: A simple data augmentation method
		for automatic speech recognition. arXiv preprint arXiv:1904.08779.

		Args:
			time_drop_width: int
			time_stripes_num: int
			freq_drop_width: int
			freq_stripes_num: int
		"""
		super(SpecAugmentation, self).__init__()

		self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num)
		self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num)

	def forward(self, input):
		x = self.time_dropper(input)
		x = self.freq_dropper(x)
		return x
