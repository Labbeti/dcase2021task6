
from torch.nn import Module, Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample

from aac.models.pann import Wavegram
from aac.transforms.cplx import CplxMelSpectrogram, CplxAmplitudeToDB
from aac.transforms.utils import Squeeze


def get_audio_transform(orig_sample_rate: int, resample_rate: int, name: str = 'raw', **kwargs) -> Module:
	assert 'sample_rate' not in kwargs.keys() or resample_rate == kwargs['sample_rate'], f'Invalid resample to {resample_rate} for audio transform with {kwargs["sample_rate"]}.'
	
	transforms = []
	if orig_sample_rate != resample_rate:
		transforms.append(Resample(orig_sample_rate, resample_rate))

	if name in ['raw', 'none', 'identity']:
		pass

	elif name in ['MelSpecDB']:
		transforms += [
			MelSpectrogram(**kwargs),
			AmplitudeToDB(),
		]

	elif name in ['CplxMelSpecDB']:
		transforms += [
			CplxMelSpectrogram(**kwargs),
			CplxAmplitudeToDB(),
		]

	elif name in ['Wavegram']:
		transforms += [
			Wavegram(**kwargs),
		]

	else:
		raise RuntimeError(f'Unknown spec transform "{name}". Must be "none", "MelSpecDB", "CplxMelSpecDB" or "Wavegram".')

	transforms.append(Squeeze())

	return Sequential(*transforms)
