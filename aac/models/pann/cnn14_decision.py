
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from aac.models.pann.models import Spectrogram, LogmelFilterBank, SpecAugmentation, ConvBlock, AttBlock, init_bn, init_layer
from aac.models.pann.utils import do_mixup, interpolate, pad_framewise_output
from aac.transforms.augments.cutoutspec import CutOutSpec
from aac.transforms.augments.mixup import Mixup


class Cnn14_DecisionLevelAtt(nn.Module):
	def __init__(
		self, 
		sample_rate: int = 32000, 
		window_size: int = 1024, 
		hop_size: int = 320, 
		mel_bins: int = 64, 
		fmin: int = 50, 
		fmax: int = 14000, 
		classes_num: int = 527,
		use_cutout: bool = False,
		use_mixup: bool = False,
		use_spec_augment: bool = False,
		return_clipwise_output: bool = True,
	) -> None:
		super().__init__()
		self.use_cutout = use_cutout
		self.use_mixup = use_mixup
		self.use_spec_augment = use_spec_augment
		self.return_clipwise_output = return_clipwise_output

		window = 'hann'
		center = True
		pad_mode = 'reflect'
		ref = 1.0
		amin = 1e-10
		top_db = None
		self.interpolate_ratio = 32	 # Downsampled ratio

		# Spectrogram extractor
		self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
			win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
			freeze_parameters=True)

		# Logmel feature extractor
		self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
			n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
			freeze_parameters=True)

		# Spec augmenter
		self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
			freq_drop_width=8, freq_stripes_num=2)
		
		self.cutout = CutOutSpec(fill_value=float(fmin))
		self.mixup = Mixup(alpha=0.4, asymmetric=True)

		self.bn0 = nn.BatchNorm2d(64)

		self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
		self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
		self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
		self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
		self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
		self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

		self.fc1 = nn.Linear(2048, 2048, bias=True)
		if self.return_clipwise_output:
			self.att_block = AttBlock(2048, classes_num, activation='sigmoid')
		else:
			self.att_block = None
		
		self.init_weight()

	def init_weight(self) -> None:
		init_bn(self.bn0)
		init_layer(self.fc1)
 
	def forward(self, input: Tensor) -> dict[str, Tensor]:
		"""
		Input: (batch_size, data_length)"""

		x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
		x = self.logmel_extractor(x)	# (batch_size, 1, time_steps, mel_bins)

		frames_num = x.shape[2]
		
		x = x.transpose(1, 3)
		x = self.bn0(x)
		x = x.transpose(1, 3)
		
		if self.training and self.use_spec_augment:
			x = self.spec_augmenter(x)
		
		if self.training and self.use_cutout:
			x = self.cutout(x)
		
		if self.training and self.use_mixup:
			mixup_lambda = self.mixup.sample_lambda()
			indexes = torch.randperm(len(x))
			x = x * mixup_lambda + x[indexes] * (1.0 - mixup_lambda)
			# x = do_mixup(x, mixup_lambda)
		
		x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
		x = F.dropout(x, p=0.2, training=self.training)
		x = torch.mean(x, dim=3)
		
		x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
		x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
		x = x1 + x2
		x = F.dropout(x, p=0.5, training=self.training)
		x = x.transpose(1, 2)
		x = F.relu_(self.fc1(x))
		x = x.transpose(1, 2)
		x = F.dropout(x, p=0.5, training=self.training)

		framewise_embed = x

		if self.return_clipwise_output:
			(clipwise_output, _, segmentwise_output) = self.att_block(x)
			segmentwise_output = segmentwise_output.transpose(1, 2)

			# Get framewise output
			framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
			framewise_output = pad_framewise_output(framewise_output, frames_num)
		else:
			clipwise_output = None
			framewise_output = None
		
		output_dict = {
			# (bsize, embed=2048, n_frames)
			'framewise_embed': framewise_embed,
			# (bsize, n_frames, n_classes)
			'framewise_output': framewise_output, 
			# (bsize, n_classes)
			'clipwise_output': clipwise_output,
		}

		return output_dict
