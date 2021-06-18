
import re

from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Any, Dict, Optional


class CustomModelCheckpoint(ModelCheckpoint):
	""" Custom Model Checkpoint with checkpoint names use '_' instead of '=' for separate name and values in checkpoint names. 
		It help for avoiding errors with hydra which uses '=' between option and values.

		Example :
			With ModelCheckpoint :
				epoch=0-step=479.ckpt
			With CustomModelCheckpoint :
				epoch_0-step_479.ckpt
	"""
	CHECKPOINT_SEP_CHAR: str = '_'

	@classmethod
	def _format_checkpoint_name(
		cls,
		filename: Optional[str],
		metrics: Dict[str, Any],
		prefix: str = "",
		auto_insert_metric_name: bool = True
	) -> str:
		if not filename:
			# filename is not set, use default name
			filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

		# check and parse user passed keys in the string
		groups = re.findall(r"(\{.*?)[:\}]", filename)
		if len(groups) >= 0:
			for group in groups:
				name = group[1:]

				if auto_insert_metric_name:
					# filename = filename.replace(group, name + "={" + name)
					filename = filename.replace(group, name + cls.CHECKPOINT_SEP_CHAR + "{" + name)

				if name not in metrics:
					metrics[name] = 0
			filename = filename.format(**metrics)

		if prefix:
			filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

		return filename
