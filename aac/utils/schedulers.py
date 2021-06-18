
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from typing import Union


def get_scheduler(name: str, optimizer: Optimizer, **kwargs) -> Union[LambdaLR, MultiStepLR, None]:
	name = str(name).lower()

	if name == 'cos_decay':
		epochs = kwargs['epochs']
		n_steps = max(epochs, 1)
		scheduler = LambdaLR(optimizer, CosDecayRule(n_steps))
	
	elif name == 'trf':
		d_model = kwargs['d_model']
		warmup_steps = kwargs['warmup_steps']		
		scheduler = LambdaLR(optimizer, TrfRule(d_model, warmup_steps))

	elif name == 'multisteplr':
		milestones = kwargs['milestones']
		gamma = kwargs['gamma']

		scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

	elif name == 'none':
		scheduler = None
	
	else:
		raise RuntimeError(f'Unknown scheduler "{name}". Must be one of ("cos_decay", "trf", "multisteplr", "none").')

	return scheduler


class CosDecayRule:
	# Note : use class for scheduler rules for being pickable for multiple-GPU with Lightning
	def __init__(self, n_steps: int) -> None:
		super().__init__()
		self.n_steps = n_steps

	def __call__(self, step: int) -> float:
		return 0.5 * (1.0 + math.cos(step * math.pi / self.n_steps))


class TrfRule:
	# Note : use class for scheduler rules for being pickable for multiple-GPU with Lightning
	def __init__(self, d_model: int, warmup_steps: int) -> None:
		super().__init__()
		self.d_model = d_model
		self.warmup_steps = warmup_steps

	def __call__(self, step: int) -> float:
		return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
