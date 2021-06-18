
from pytorch_lightning import LightningDataModule
from torch.nn import Sequential
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from typing import Any, Optional, Union

from aac.datasets.clotho import Clotho
from aac.transforms.captions import WordsToIdx, Tokenizer
from aac.transforms.collate_fn import PadCollate
from aac.transforms.get import get_audio_transform
from aac.utils.vocabulary import IGNORE_ID, compute_word_freqs


class ClothoDataModule(LightningDataModule):
	def __init__(
		self,
		root: str = '../data',
		download: bool = False,
		version: str = 'v2.1',
		bsize: int = 32,
		n_workers: int = 1,
		verbose: bool = True,
		add_sos_and_eos: bool = False,
		train_captions_select: Union[str, int] = 'random',
		resample_rate: int = 32000,
		audio_params: Optional[dict[str, Any]] = None,
	) -> None:
		if audio_params is None:
			audio_params = {}
		super().__init__()
		self._root = root
		self._download = download
		self._version = version
		self._bsize = bsize
		self._n_workers = n_workers
		self._verbose = verbose
		self._add_sos_and_eos = add_sos_and_eos
		self._train_captions_select = train_captions_select
		self._resample_rate = resample_rate
		self._audio_params = audio_params

		# Init in prepare_data
		self._audio_transform = None
		self._captions_transform = None
		self._word_freqs = None
		self._idx_to_word = None
		self._word_to_idx = None
		# Init in setup
		self._dev_clotho = None
		self._val_clotho = None
		self._eval_clotho = None
		self._test_clotho = None

		self._audio_pad = 0.0
		self._caption_pad = IGNORE_ID
		
		common_params_collate = dict(audio_fill=self._audio_pad, caption_fill=self._caption_pad)
		self._train_collate = PadCollate(**common_params_collate, captions_select=self._train_captions_select, sort_batch=True)
		self._val_collate = PadCollate(**common_params_collate, captions_select=self._train_captions_select, sort_batch=True)
		self._test_collate = PadCollate(**common_params_collate, captions_select='all', sort_batch=False)

	def prepare_data(self) -> None:
		if self._download:
			dataset_params = dict(download=True, version=self._version)
			_ = Clotho(subset='dev', root=self._root, **dataset_params)
			_ = Clotho(subset='eval', root=self._root, **dataset_params)
			_ = Clotho(subset='test', root=self._root, **dataset_params)
		
		# Build audio transform
		self._audio_transform = get_audio_transform(
			orig_sample_rate=Clotho.SAMPLE_RATE, 
			resample_rate=self._resample_rate,
			**self._audio_params,
		)

		# Build vocabulary and captions transform
		datasets = [Clotho(root=self._root, subset='dev', version=self._version)]
		tokenizer = Tokenizer(add_sos_and_eos=self._add_sos_and_eos)
		word_freqs = compute_word_freqs(datasets, tokenizer, verbose=self._verbose)
		
		self._word_freqs = word_freqs
		self._idx_to_word = dict(enumerate(self._word_freqs.keys()))
		self._word_to_idx = {word: idx for idx, word in self._idx_to_word.items()}
		
		self._captions_transform = Sequential(
			tokenizer,
			WordsToIdx(self._word_to_idx),
		)

	def setup(self, stage: Optional[str] = None) -> None:
		dataset_params = dict(
			download=False,
			version=self._version,
			audio_cache=True,
			audio_transform=self._audio_transform,
			captions_transform=self._captions_transform,
		)

		if self._dev_clotho is None:
			self._dev_clotho = Clotho(
				root=self._root, 
				subset='dev',
				**dataset_params,
			)

		if self._version != 'v1':
			if self._val_clotho is None:
				self._val_clotho = Clotho(
					root=self._root, 
					subset='val',
					**dataset_params,
				)

		if stage in ['test', None]:
			if self._eval_clotho is None:
				self._eval_clotho = Clotho(
					root=self._root, 
					subset='eval',
					**dataset_params,
				)

			if self._test_clotho is None:
				self._test_clotho = Clotho(
					root=self._root, 
					subset='test',
					**dataset_params,
				)

	def train_dataloader(self) -> DataLoader:
		assert self._dev_clotho is not None
		return DataLoader(
			dataset=self._dev_clotho,
			batch_size=self._bsize,
			num_workers=self._n_workers,
			shuffle=True,
			collate_fn=self._train_collate,
			pin_memory=True,
			drop_last=False,
		)

	def val_dataloader(self) -> DataLoader:
		if self._version == 'v1':
			dataset = self._eval_clotho
		else:
			dataset = self._val_clotho
		assert dataset is not None, 'Clotho val dataset is None.'
		return DataLoader(
			dataset=dataset,
			batch_size=self._bsize,
			num_workers=self._n_workers,
			shuffle=False,
			collate_fn=self._val_collate,
			pin_memory=True,
			drop_last=False,
		)

	def test_dataloader(self) -> list[DataLoader]:
		common_params = dict(
			batch_size=1,
			num_workers=self._n_workers,
			shuffle=False,
			collate_fn=self._test_collate,
			pin_memory=True,
			drop_last=False,
		)
		return [
			DataLoader(dataset=dataset, **common_params) 
			for dataset in self.test_datasets
		]

	@property
	def test_datasets(self) -> list[Dataset]:
		if self._version == 'v1':
			datasets = [self._dev_clotho, self._eval_clotho, self._test_clotho]
		else:
			datasets = [self._dev_clotho, self._val_clotho, self._eval_clotho, self._test_clotho]
		assert all((dataset is not None for dataset in datasets)), 'Datasets are not initialized for testing.'
		return datasets
	
	@property
	def hparams(self) -> dict[str, Any]:
		return {
			'version': self._version,
			'bsize': self._bsize,
			'add_sos_and_eos': self._add_sos_and_eos,
			'train_captions_select': self._train_captions_select,
			'resample_rate': self._resample_rate,
			'audio_params': self._audio_params,
		}

	@property
	def vocabulary(self) -> dict[str, None]:
		assert self._word_freqs is not None
		return dict.fromkeys(self._word_freqs.keys())

	@property
	def idx_to_word(self) -> dict[int, str]:
		assert self._idx_to_word is not None
		return self._idx_to_word

	@property
	def word_to_idx(self) -> dict[str, int]:
		assert self._word_to_idx is not None
		return self._word_to_idx
