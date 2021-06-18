
import numpy as np
import random
import subprocess
import torch

from torch.nn import Module
from typing import Iterable, Optional, Union


def reset_seed(seed: Optional[int]):
	"""
		Reset the seed of following packages for reproductibility :
			- random
			- numpy
			- torch
			- torch.cuda

		Also set deterministic behaviour for cudnn backend.

		:param seed: The seed to set.
	"""
	if seed is not None:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

		if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
		else:
			raise RuntimeError(
				'Cannot make deterministic behaviour for current torch backend (cannot find "torch.backends.cudnn" submodule).'
			)


def get_current_git_hash() -> str:
	"""
		Return the current git hash in the current directory.

		:returns: The git hash. If an error occurs, returns 'UNKNOWN'.
	"""
	try:
		git_hash = subprocess.check_output(['git', 'describe', '--always'])
		git_hash = git_hash.decode('UTF-8').replace('\n', '')
		return git_hash
	except subprocess.CalledProcessError:
		return 'UNKNOWN'


def count_params(model: Module, only_trainable: bool = False) -> int:
    return sum((param.numel() for param in model.parameters() if not only_trainable or param.requires_grad))


def checksum(model: Module, only_trainable: bool = False) -> float:
    return sum((float(param.sum().item()) for param in model.parameters() if not only_trainable or param.requires_grad))


def split_indexes(indexes: Iterable[int], ratios: Iterable[float], shuffle: bool = False) -> list[list[int]]:
	indexes = list(indexes)
	ratios = list(ratios)
	assert 0.0 <= sum(ratios) <= 1.0 + 1e-10, 'Ratio sum must be in range [0, 1].'

	if shuffle:
		random.shuffle(indexes)

	prev_idx = 0
	split = []
	for ratio in ratios:
		next_idx = prev_idx + round(len(indexes) * ratio)
		split.append(indexes[prev_idx:next_idx])
		prev_idx = next_idx
	return split


def formatted_duration(duration: Union[int, float]) -> str:
	""" Get formatted duration elapsed : HH:mm:ss """
	duration = int(duration)
	rest, seconds = divmod(duration, 60)
	hours, minutes = divmod(rest, 60)
	duration_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
	return duration_str
