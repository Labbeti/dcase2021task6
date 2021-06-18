
from pytorch_lightning import LightningDataModule
from typing import Optional

from .clotho import ClothoDataModule


def get_datamodule(name: str, **kwargs) -> Optional[LightningDataModule]:
	if name in ['Clotho']:
		datamodule = ClothoDataModule(**kwargs)
	elif name in ['none']:
		datamodule = None
	else:
		raise RuntimeError(f'Unknown datamodule name "{name}".')
	return datamodule
