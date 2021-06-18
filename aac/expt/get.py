
from pytorch_lightning import LightningModule

from .cnnt import LMCnnTell
from .lat import LMListenAttendTell


def get_exptmodule(name: str, **kwargs) -> LightningModule:
	name = str(name).lower()
	if name in ['listenattendtell', 'lat']:
		exptmodule = LMListenAttendTell(**kwargs)
	elif name in ['cnntell', 'cnnt']:
		exptmodule = LMCnnTell(**kwargs)
	else:
		raise RuntimeError(f'Unknown experiment "{name}".')
	return exptmodule
