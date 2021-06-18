
import torch

from torch import Tensor
from torch.nn import Module, LogSoftmax
from typing import Optional


def identity(x):
	return x


class NLLLossVecTarget(Module):
	def __init__(self, dim: int = -1, reduction: str = 'mean', ignore_index: Optional[int] = None):
		super().__init__()
		self._dim = dim
		self._reduction = reduction
		self._ignore_index = ignore_index

		if reduction == 'mean':
			self._reduce_fn = torch.mean
		elif reduction == 'sum':
			self._reduce_fn = torch.sum
		elif reduction == 'none':
			self._reduce_fn = identity
		else:
			raise RuntimeError(f'Invalid reduction "{self._reduction}". Must be "mean", "sum" or "none".')

	def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
		assert log_probs.shape == target.shape

		if self._ignore_index is None:
			loss = - torch.sum(log_probs * target, dim=self._dim)
			loss = self._reduce_fn(loss)
		else:
			loss = log_probs * target
			loss[target == self._ignore_index] = 0
			loss = - torch.sum(loss, dim=self._dim)

			if self._reduction == 'mean':
				loss = torch.sum(loss)
				n_tokens = (target != self._ignore_index).sum()
				loss = loss / n_tokens
			else:
				loss = self._reduce_fn(loss)

		return loss


class CrossEntropyVecTarget(Module):
	def __init__(self, dim: int = -1, reduction: str = 'mean', ignore_index: Optional[int] = None):
		super().__init__()
		self._log_softmax = LogSoftmax(dim=dim)
		self._nll_loss = NLLLossVecTarget(dim=dim, reduction=reduction, ignore_index=ignore_index)

	def forward(self, logits: Tensor, target: Tensor) -> Tensor:
		return self._nll_loss(self._log_softmax(logits), target)
