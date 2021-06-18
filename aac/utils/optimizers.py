
from torch import Tensor
from torch.optim import Optimizer, Adam
from typing import Iterator


def get_optimizer(name: str, parameters: Iterator[Tensor], **kwargs) -> Optimizer:
	if name == 'Adam':
		return Adam(parameters, **kwargs)
	else:
		raise RuntimeError(f'Unknown optimizer "{name}".')
