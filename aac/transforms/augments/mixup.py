
import torch

from torch import Tensor
from torch.nn import Module
from torch.distributions.beta import Beta


class Mixup(Module):
	def __init__(self, alpha: float = 0.4, asymmetric: bool = False):
		super().__init__()
		self._alpha = alpha
		self._asymmetric = asymmetric
		self._beta = Beta(alpha, alpha)
		self._prev_lbd = -1.0

	def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
		if len(x) != len(y):
			raise RuntimeError(f'Data to mix must have the same size along the first dim. ({len(x)} != {len(y)})')

		lambda_ = self.sample_lambda()
		indexes = torch.randperm(len(x))

		x_mix = x * lambda_ + x[indexes] * (1.0 - lambda_)
		y_mix = y * lambda_ + y[indexes] * (1.0 - lambda_)
		return x_mix, y_mix

	def sample_lambda(self) -> float:
		lambda_ = self._beta.sample()
		if self._asymmetric:
			lambda_ = max(lambda_, 1.0 - lambda_)
		self._prev_lbd = lambda_
		return lambda_
	
	def prev_lambda(self) -> float:
		return self._prev_lbd
