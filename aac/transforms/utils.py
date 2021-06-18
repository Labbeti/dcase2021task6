
import torch

from torch import Tensor
from torch.nn import Module
from typing import Callable, Optional


class Squeeze(Module):
	def __init__(self, dim: Optional[int] = None):
		super().__init__()
		self.dim = dim

	def forward(self, x: Tensor) -> Tensor:
		if self.dim is None:
			return torch.squeeze(x)
		else:
			return torch.squeeze(x, self.dim)

	def extra_repr(self) -> str:
		return f'dim={self.dim}'


class AmplitudeToLog(Module):
	def __init__(self, eps: float = torch.finfo(torch.float).eps):
		super().__init__()
		self.eps = eps

	def forward(self, data: Tensor) -> Tensor:
		return torch.log(data + self.eps)


class Normalize(Module):
	def forward(self, data: Tensor) -> Tensor:
		return data / data.abs().max()


class ModuleWrap(Module):
	def __init__(self, fn: Callable):
		super().__init__()
		self.fn = fn
	
	def forward(self, *args, **kwargs):
		return self.fn(*args, **kwargs)


class Reshape(Module):
	def __init__(self, shape: tuple[int, ...]):
		super().__init__()
		self.shape = shape

	def forward(self, x: Tensor) -> Tensor:
		return torch.reshape(x, self.shape)
