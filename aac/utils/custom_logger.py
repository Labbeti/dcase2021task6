
import os.path as osp
import time

from argparse import Namespace
from omegaconf import DictConfig
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from typing import Any, Optional, Union

from aac.utils.misc import get_current_git_hash


class CustomTensorboardLogger(TensorBoardLogger):
	"""Custom Tensorboard Logger for saving hparams and metrics in tensorboard because we cannot save hparams and metrics several times in SummaryWriter.

	Note : hparams and metrics are saved when 'save_and_close' is called.
	"""
	HPARAMS_FNAME = 'hparams.yaml'
	METRICS_FNAME = 'metrics.yaml'


	def __init__(
		self,
		save_dir: str,
		name: Optional[str] = 'default',
		version: Optional[Union[int, str]] = None,
		log_graph: bool = False,
		default_hp_metric: bool = True,
		prefix: str = '',
		params: Union[dict[str, Any], DictConfig, None] = None,
		verbose: bool = True,
		**kwargs,
	) -> None:
		super().__init__(
			save_dir=save_dir,
			name=name,
			version=version,
			log_graph=log_graph,
			default_hp_metric=default_hp_metric,
			prefix=prefix,
			**kwargs,
		)

		params = _convert_dict_like_to_dict(params)
		if default_hp_metric:
			metrics = {'hp_metric': -1}
		else:
			metrics = {}

		self._all_hparams = params
		self._all_metrics = metrics
		self._verbose = verbose

		self._start_time = time.time()
		self._closed = False
	
	def log_hyperparams(
		self,
		params: Union[dict[str, Any], None, Namespace],
		metrics: Union[dict[str, Any], None] = None,
	) -> None:
		params = _convert_dict_like_to_dict(params)
		metrics = _convert_dict_like_to_dict(metrics)

		self._all_hparams.update(params)
		self._all_metrics.update(metrics)
		self.experiment.flush()

	def finalize(self, status: str) -> None:
		# Called at the end of the training (after trainer.fit())
		self.experiment.flush()

	def save_and_close(self) -> None:
		if self._closed:
			raise RuntimeError('CustomTensorboardLogger cannot be closed twice.')

		prefix = f'{self.name}_{self.version}'
		duration = self._get_duration()
		self._all_hparams['duration'] = duration
		self.experiment.add_text(f'{prefix}/duration', duration)

		git_hash = get_current_git_hash()
		self._all_hparams['git_hash'] = git_hash
		self.experiment.add_text(f'{prefix}/git_hash', git_hash)

		self._all_hparams = {k: _convert_value(v) for k, v in self._all_hparams.items()}
		self._all_metrics = {k: _convert_value(v) for k, v in self._all_metrics.items()}

		self._all_hparams = dict(sorted(self._all_hparams.items()))
		# self._all_metrics = dict(sorted(self._all_metrics.items()))

		self.experiment.add_text(f'{prefix}/all_hparams', str(self._all_hparams))
		self.experiment.add_text(f'{prefix}/all_metrics', str(self._all_metrics))

		fpath_hparams = osp.join(self.log_dir, self.HPARAMS_FNAME)
		save_hparams_to_yaml(fpath_hparams,  self._all_hparams)

		fpath_metrics = osp.join(self.log_dir, self.METRICS_FNAME)
		save_hparams_to_yaml(fpath_metrics,  self._all_metrics)

		super().log_hyperparams(self._all_hparams, self._all_metrics)
		self.experiment.flush()

		self._closed = True

	def _get_duration(self) -> str:
		# Get formatted duration elapsed : HH:mm:ss
		duration = int(time.time() - self._start_time)
		rest, seconds = divmod(duration, 60)
		hours, minutes = divmod(rest, 60)
		duration_str = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
		return duration_str

	def is_closed(self) -> bool:
		return self._closed

	@property
	def hparams(self) -> dict:
		return self._all_hparams

	@hparams.setter
	def hparams(self, other: dict) -> None:
		self._all_hparams = other

	@property
	def metrics(self) -> dict:
		return self._all_metrics


def _convert_value(v: Any) -> Any:
	if isinstance(v, Tensor):
		return v.item()
	elif isinstance(v, bool):
		return str(v)
	else:
		return v


def _convert_dict_like_to_dict(dic: Union[dict, Namespace, DictConfig, None]) -> dict:
	if isinstance(dic, (DictConfig, Namespace)):
		return dic.__dict__
	elif dic is None:
		return {}
	else:
		return dic
