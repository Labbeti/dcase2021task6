
import math
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torchaudio.functional import create_fb_matrix
from torchaudio.transforms import MelScale
from typing import Callable, Optional


def cplx_spectrogram(
	waveform: Tensor,
	pad: int,
	window: Tensor,
	n_fft: int,
	hop_length: int,
	win_length: int,
	power: Optional[float],
	normalized: bool,
	center: bool = True,
	pad_mode: str = "reflect",
	onesided: bool = True
) -> Tensor:
	r"""
		Based on torchaudio 0.8.0 available at https://pytorch.org/audio/stable/_modules/torchaudio/functional/functional.html#spectrogram
	"""

	if pad > 0:
		waveform = F.pad(waveform, (pad, pad), "constant")

	# pack batch
	shape = waveform.size()
	waveform = waveform.reshape(-1, shape[-1])

	# default values are consistent with librosa.core.spectrum._spectrogram
	spec_f = torch.stft(
		input=waveform,
		n_fft=n_fft,
		hop_length=hop_length,
		win_length=win_length,
		window=window,
		center=center,
		pad_mode=pad_mode,
		normalized=False,
		onesided=onesided,
		return_complex=True,
	)

	# unpack batch
	spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

	if normalized:
		spec_f /= window.pow(2.).sum().sqrt()
	
	if power is None or power == 1.0:
		return spec_f
	else:
		return spec_f.pow(power)


class CplxSpectrogram(Module):	
	def __init__(
		self,
		n_fft: int = 400,
		win_length: Optional[int] = None,
		hop_length: Optional[int] = None,
		pad: int = 0,
		window_fn: Callable[..., Tensor] = torch.hann_window,
		power: Optional[float] = 2.,
		normalized: bool = False,
		wkwargs: Optional[dict] = None,
		center: bool = True,
		pad_mode: str = "reflect",
		onesided: bool = True,
	) -> None:

		super().__init__()
		self.n_fft = n_fft
		self.win_length = win_length if win_length is not None else n_fft
		self.hop_length = hop_length if hop_length is not None else self.win_length // 2
		window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
		self.register_buffer('window', window)
		self.pad = pad
		self.power = power
		self.normalized = normalized
		self.center = center
		self.pad_mode = pad_mode
		self.onesided = onesided

	def forward(self, waveform: Tensor) -> Tensor:
		return cplx_spectrogram(
			waveform,
			self.pad,
			self.window,
			self.n_fft,
			self.hop_length,
			self.win_length,
			self.power,
			self.normalized,
			self.center,
			self.pad_mode,
			self.onesided
		)


class CplxMelScale(Module):
	__constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

	def __init__(self,
				 n_mels: int = 128,
				 sample_rate: int = 16000,
				 f_min: float = 0.,
				 f_max: Optional[float] = None,
				 n_stft: Optional[int] = None,
				 norm: Optional[str] = None) -> None:
		super().__init__()
		self.n_mels = n_mels
		self.sample_rate = sample_rate
		self.f_max = f_max if f_max is not None else float(sample_rate // 2)
		self.f_min = f_min
		self.norm = norm

		assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)

		fb = torch.empty(0) if n_stft is None else create_fb_matrix(
			n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm)
		self.register_buffer('fb', fb)

	def forward(self, specgram: Tensor) -> Tensor:
		# pack batch
		shape = specgram.size()
		specgram = specgram.reshape(-1, shape[-2], shape[-1])

		if self.fb.numel() == 0:
			tmp_fb = F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max,
										self.n_mels, self.sample_rate, self.norm)
			# Attributes cannot be reassigned outside __init__ so workaround
			self.fb.resize_(tmp_fb.size())
			self.fb.copy_(tmp_fb)

		# (channel, frequency, time).transpose(...) dot (frequency, n_mels)
		# -> (channel, time, n_mels).transpose(...)
		fb = torch.complex(self.fb, self.fb)
		mel_specgram = torch.matmul(specgram.transpose(1, 2), fb).transpose(1, 2)

		# unpack batch
		mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

		return mel_specgram


class CplxMelSpectrogram(Module):
	r"""
		Based on torchaudio MelSpectrogram code : https://pytorch.org/audio/stable/_modules/torchaudio/transforms.html#MelSpectrogram
	"""
	__constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

	def __init__(self,
				 sample_rate: int = 16000,
				 n_fft: int = 400,
				 win_length: Optional[int] = None,
				 hop_length: Optional[int] = None,
				 f_min: float = 0.,
				 f_max: Optional[float] = None,
				 pad: int = 0,
				 n_mels: int = 128,
				 window_fn: Callable[..., Tensor] = torch.hann_window,
				 power: Optional[float] = 2.,
				 normalized: bool = False,
				 wkwargs: Optional[dict] = None,
				 center: bool = True,
				 pad_mode: str = "reflect",
				 onesided: bool = True,
				 norm: Optional[str] = None) -> None:
		super().__init__()
		self.sample_rate = sample_rate
		self.n_fft = n_fft
		self.win_length = win_length if win_length is not None else n_fft
		self.hop_length = hop_length if hop_length is not None else self.win_length // 2
		self.pad = pad
		self.power = power
		self.normalized = normalized
		self.n_mels = n_mels  # number of mel frequency bins
		self.f_max = f_max
		self.f_min = f_min
		self.spectrogram = CplxSpectrogram(n_fft=self.n_fft, win_length=self.win_length,
									   hop_length=self.hop_length,
									   pad=self.pad, window_fn=window_fn, power=self.power,
									   normalized=self.normalized, wkwargs=wkwargs,
									   center=center, pad_mode=pad_mode, onesided=onesided)
		self.mel_scale = CplxMelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, norm)

	def forward(self, waveform: Tensor) -> Tensor:
		specgram = self.spectrogram(waveform)
		mel_specgram = self.mel_scale(specgram)
		return mel_specgram


class CplxAmplitudeToDB(Module):
	def __init__(self, stype: str = 'power', top_db: Optional[float] = None) -> None:
		super().__init__()
		self.stype = stype
		self.multiplier = 10.0 if stype == 'power' else 20.0
		self.amin = 1e-10
		self.ref_value = 1.0
		self.db_multiplier = math.log10(max(self.amin, self.ref_value))

	def forward(self, x: Tensor) -> Tensor:
		x = torch.complex(torch.clamp(x.real, min=self.amin), torch.clamp(x.imag, min=self.amin)) 
		x_db = self.multiplier * torch.log10(x)
		x_db -= self.multiplier * self.db_multiplier
		return x_db
