
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from torch.optim import Optimizer
from typing import Any


class LogLRCallback(Callback):
	def __init__(self, prefix: str = 'train/'):
		super().__init__()
		self.prefix = prefix

	def on_train_batch_end(self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
		self._log_impl(pl_module, on_epoch=False, on_step=True)

	def _log_impl(self, pl_module: LightningModule, **kwargs):
		optimizer = pl_module.optimizers()

		if isinstance(optimizer, Optimizer):
			learning_rates = [param_group['lr'] for param_group in optimizer.param_groups]

			if len(learning_rates) == 1:
				pl_module.log(f'{self.prefix}lr', learning_rates[0], **kwargs)
			else:
				for i, lr in enumerate(learning_rates):
					pl_module.log(f'{self.prefix}lr{i}', lr, **kwargs)
		else:
			raise RuntimeError(f'Unsupported optimizer type "{str(type(optimizer))}".')
