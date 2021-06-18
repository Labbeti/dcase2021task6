import csv
import logging
import os
import os.path as osp
import torchaudio

from py7zr import SevenZipFile
from torch import Tensor
from torch.nn import Module
from torch.utils.data.dataset import Dataset
from torchaudio.datasets.utils import download_url
from typing import Any, Optional


class Clotho(Dataset):
	"""
		Unofficial Clotho pytorch dataset for DCASE 2021 Task 6.
		Subsets available are 'train', 'val', 'eval' and 'test' (for v2 and v2.1).

		Audio are waveform sounds of 15 to 30 seconds, sampled at 44100 Hz.
		Target is a list of 5 different sentences strings describing an audio sample.
		Sentence max number of word is 20.

		Clotho V1 Paper : https://arxiv.org/pdf/1910.09387.pdf

		Dataset folder tree for version 'v2.1': 

		```
		root/
		└── CLOTHO_{version}
			├── clotho_audio_files
			│	├── development
			│	│	└── (3840 files, ~8.5G)
			│	├── evaluation
			│	│	└── (1045 files, ~2.4G)
			│	├── validation
			│	│	└── (1046 files, ~2.4G)
			│	└── test
			│		└── (1044 files, ~2.4G)
			└── clotho_csv_files
				├── clotho_captions_development.csv
				├── clotho_captions_evaluation.csv
				├── clotho_captions_validation.csv
				├── clotho_metadata_development.csv
				├── clotho_metadata_evaluation.csv
				├── clotho_metadata_test.csv
				└── clotho_metadata_validation.csv
		```
	"""

	SAMPLE_RATE: int = 44100
	AUDIO_MIN_LENGTH: int = 15  # in seconds
	AUDIO_MAX_LENGTH: int = 30  # in seconds

	def __init__(
		self,
		root: str,
		subset: str,
		download: bool = False,
		audio_transform: Optional[Module] = None,
		captions_transform: Optional[Module] = None,
		audio_cache: bool = False,
		version: str = 'v2.1',
		verbose: int = 0,
	) -> None:
		"""
			:param root: The parent of the dataset root directory. The data will be stored in the 'CLOTHO_{VERSION}' subdirectory.
			:param subset: The subset of Clotho to use. Can be 'dev', 'val', 'eval' or 'test'.
			:param download: Download the dataset if download=True and if the dataset is not already downloaded.
				(default: False)
			:param audio_transform: The transform to apply to waveforms (Tensor).
				(default: None)
			:param captions_transform: The transform to apply to captions with a list of 5 sentences (List[str]).
				(default: None)
			:param audio_cache: If True, store audio waveforms into RAM memory after loading them from files.
				Can increase the data loading process time performance but requires enough RAM to store the data.
				(default: False)
			:param version: The version of the dataset. Can be 'v1', 'v2' or 'v2.1'.
				(default: 'v2.1')
			:param verbose: Verbose level to use. Can be 0 or 1.
				(default: 0)
		"""
		if version not in CLOTHO_LINKS.keys():
			raise RuntimeError(f'Invalid Clotho version. Must be one of {tuple(CLOTHO_LINKS.keys())}.')

		if subset not in CLOTHO_LINKS[version].keys():
			raise RuntimeError(f'Invalid Clotho subset for version {version}. Must be one of {tuple(CLOTHO_LINKS[version].keys())}.')

		super().__init__()
		self._root = root
		self._subset = subset
		self._download = download
		self._audio_transform = audio_transform
		self._captions_transform = captions_transform
		self._audio_cache = audio_cache
		self._version = version
		self._verbose = verbose

		self._data_info = {}
		self._idx_to_fname = []
		self._waveforms = {}

		if self._download:
			self._prepare_data()
		self._load_data()

	def __getitem__(self, index: int) -> tuple[Tensor, list[str]]:
		"""
			Get the audio data as 1D tensor and the matching captions as 5 sentences.

			:param index: The index of the item.
			:return: A tuple of audio data of shape (size,) and the 5 matching captions.
		"""
		audio = self.get_audio(index)
		captions = self.get_captions(index)
		return audio, captions

	def __len__(self) -> int:
		"""
			:return: The number of items in the dataset.
		"""
		return len(self._data_info)

	def get_audio(self, index: int) -> Tensor:
		"""
			:param index: The index of the item.
			:return: The audio data as 1D tensor.
		"""
		if not self._audio_cache or index not in self._waveforms.keys():
			fpath = self.get_audio_fpath(index)
			audio, sample_rate = torchaudio.load(fpath)
			
			if sample_rate != self.SAMPLE_RATE:
				raise RuntimeError(f'Invalid sample rate {sample_rate}Hz of audio file {fpath} with Clotho {self.SAMPLE_RATE}Hz.')

			if self._audio_cache:
				self._waveforms[index] = audio

		else:
			audio = self._waveforms[index]

		if self._audio_transform is not None:
			audio = self._audio_transform(audio)
		return audio

	def get_captions(self, index: int) -> list[str]:
		"""
			:param index: The index of the item.
			:return: The list of 5 captions of an item.
		"""
		info = self._data_info[self.get_audio_fname(index)]
		captions = info['captions']

		if self._captions_transform is not None:
			captions = self._captions_transform(captions)
		return captions

	def get_metadata(self, index: int) -> dict[str, str]:
		"""
			Returns the metadata dictionary for a file.
			This dictionary contains :
				- 'keywords': Contains the keywords of the item, separated by ';'.
				- 'sound_id': Id of the audio.
				- 'sound_link': Link to the audio.
				- 'start_end_samples': The range of the sound where it was extracted.
				- 'manufacturer': The manufacturer of this file.
				- 'licence': Link to the licence.

			:param index: The index of the item.
			:return: The metadata dictionary associated to the item.
		"""
		info = self._data_info[self.get_audio_fname(index)]
		return info['metadata']

	def get_audio_fname(self, index: int) -> str:
		"""
			:param index: The index of the item.
			:return: The fname associated to the index.
		"""
		return self._idx_to_fname[index]

	def get_audio_fpath(self, index: int) -> str:
		"""
			:param index: The index of the item.
			:return: The fpath associated to the index.
		"""
		info = self._data_info[self.get_audio_fname(index)]
		return info['fpath']

	def get_audio_cache(self) -> dict[int, Tensor]:
		"""
			:return: Returns the audio cache of the audio waveforms.
		"""
		return self._waveforms
	
	@property
	def subset(self) -> str:
		return self._subset
	
	@property
	def _dpath_audio(self) -> str:
		return osp.join(self._root, f'CLOTHO_{self._version}', 'clotho_audio_files')
	
	@property
	def _dpath_csv(self) -> str:
		return osp.join(self._root, f'CLOTHO_{self._version}', 'clotho_csv_files')

	def _prepare_data(self) -> None:
		if self._verbose >= 1:
			logging.info(f'Download files for the dataset "{self.__class__.__name__}", subset "{self._subset}"...')

		if not osp.isdir(self._root):
			os.mkdir(self._root)
		
		if not osp.isdir(self._dpath_audio):
			os.makedirs(self._dpath_audio)
		
		if not osp.isdir(self._dpath_csv):
			os.makedirs(self._dpath_csv)

		infos = CLOTHO_LINKS[self._version][self._subset]

		# Download files
		for _, info in infos.items():
			fname, url, hash_ = info['fname'], info['url'], info['hash']
			extension = fname.split('.')[-1]

			if extension == '7z':
				dpath = self._dpath_audio
			elif extension == 'csv':
				dpath = self._dpath_csv
			else:
				raise RuntimeError(f'Invalid extension "{extension}". Must be "7z" or "csv".')

			fpath = osp.join(dpath, fname)

			if not osp.isfile(fpath):
				if osp.exists(fpath):
					raise RuntimeError(f'Object "{fpath}" already exists but it is not a file.')

				if self._verbose >= 1:
					logging.info(f'Download file "{fname}" from url "{url}"...')

				try:
					download_url(url, dpath, fname, hash_value=hash_, hash_type='md5', progress_bar=self._verbose >= 1)
				except KeyboardInterrupt as err:
					os.remove(fpath)
					raise err

		# Extract audio files from archives
		for _, info in infos.items():
			fname = info['fname']
			extension = fname.split('.')[-1]

			if extension == '7z':
				fpath = osp.join(self._dpath_audio, fname)
				extracted_dpath = osp.join(self._dpath_audio, CLOTHO_AUDIO_DNAMES[self._subset])

				if not osp.isdir(extracted_dpath):
					if self._verbose >= 1:
						logging.info(f'Extract archive file "{fname}"...')
					
					with SevenZipFile(fpath) as file:
						file.extractall(self._dpath_audio)

				elif self._verbose >= 1:
					logging.info(f'Archive "{fname}" already extracted. (extracted dpath "{extracted_dpath}" already exists)')

			elif extension == 'csv':
				pass
			
			else:
				logging.error(f'Found unexpected extension "{extension}" for downloaded file "{fname}". Must be "7z" or "csv".')

	def _load_data(self) -> None:
		# Read fpath of .wav audio files
		infos = CLOTHO_LINKS[self._version][self._subset]
		dname_audio = CLOTHO_AUDIO_DNAMES[self._subset]
		dpath_audio_subset = osp.join(self._dpath_audio, dname_audio)

		self._data_info = {
			fname: {
				'fpath': osp.join(dpath_audio_subset, fname),
				'captions': [],
			}
			for fname in os.listdir(dpath_audio_subset)
		}

		# --- Read captions info
		if 'captions' in infos.keys():
			captions_fname = infos['captions']['fname']
			captions_fpath = osp.join(self._dpath_csv, captions_fname)

			with open(captions_fpath, 'r') as file:
				reader = csv.DictReader(file)
				for row in reader:
					# Keys: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
					fname = row['file_name']
					if fname in self._data_info.keys():
						self._data_info[fname]['captions'] = [
							row[caption_key] for caption_key in ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
						]
					else:
						raise RuntimeError(f'Found fname "{fname}" in CSV "{captions_fname}" but not the corresponding audio file.')

		# --- Read metadata info
		metadata_fname = infos['metadata']['fname']
		metadata_fpath = osp.join(self._dpath_csv, metadata_fname)

		# Keys: file_name, keywords, sound_id, sound_link, start_end_samples, manufacturer, license
		if self._version in ['v2', 'v2.1']:
			encoding = 'ISO-8859-1'
		else:
			encoding = None

		with open(metadata_fpath, 'r', encoding=encoding) as file:	
			delimiter = ';' if self._subset == 'test' else ','
			reader = csv.DictReader(file, delimiter=delimiter)
			
			for row in reader:
				fname = row['file_name']
				row_copy = dict(row)
				row_copy.pop('file_name')

				if fname in self._data_info.keys():
					self._data_info[fname]['metadata'] = row_copy
				else:
					raise RuntimeError(f'Found fname "{fname}" in CSV "{metadata_fname}" but not the corresponding audio file.')

		self._idx_to_fname = [fname for fname in self._data_info.keys()]


CLOTHO_LINKS = {
	'v1': {
		'dev': {
			'audio_archive': {
				'fname': 'clotho_audio_development.7z',
				'url': 'https://zenodo.org/record/3490684/files/clotho_audio_development.7z?download=1',
				'hash': 'e3ce88561b317cc3825e8c861cae1ec6',
			},
			'captions': {
				'fname': 'clotho_captions_development.csv',
				'url': 'https://zenodo.org/record/3490684/files/clotho_captions_development.csv?download=1',
				'hash': 'dd568352389f413d832add5cf604529f',
			},
			'metadata': {
				'fname': 'clotho_metadata_development.csv',
				'url': 'https://zenodo.org/record/3490684/files/clotho_metadata_development.csv?download=1',
				'hash': '582c18ee47cebdbe33dce1feeab53a56',
			},
		},
		'eval': {
			'audio_archive': {
				'fname': 'clotho_audio_evaluation.7z',
				'url': 'https://zenodo.org/record/3490684/files/clotho_audio_evaluation.7z?download=1',
				'hash': '4569624ccadf96223f19cb59fe4f849f',
			},
			'captions': {
				'fname': 'clotho_captions_evaluation.csv',
				'url': 'https://zenodo.org/record/3490684/files/clotho_captions_evaluation.csv?download=1',
				'hash': '1b16b9e57cf7bdb7f13a13802aeb57e2',
			},
			'metadata': {
				'fname': 'clotho_metadata_evaluation.csv',
				'url': 'https://zenodo.org/record/3490684/files/clotho_metadata_evaluation.csv?download=1',
				'hash': '13946f054d4e1bf48079813aac61bf77',
			},
		},
		'test': {
			'audio_archive': {
				'fname': 'clotho_audio_test.7z',
				'url': 'https://zenodo.org/record/3865658/files/clotho_audio_test.7z?download=1',
				'hash': '9b3fe72560a621641ff4351ba1154349',
			},
			'metadata': {
				'fname': 'clotho_metadata_test.csv',
				'url': 'https://zenodo.org/record/3865658/files/clotho_metadata_test.csv?download=1',
				'hash': '52f8ad01c229a310a0ff8043df480e21',
			},
		},
	},
	'v2': {
		'dev': {
			'audio_archive': {
				'fname': 'clotho_audio_development.7z',
				'url': 'https://zenodo.org/record/4743815/files/clotho_audio_development.7z?download=1',
				'hash': 'eda144a5e05a60e6d2e37a65fc4720a9',
			},
			'captions': {
				'fname': 'clotho_captions_development.csv',
				'url': 'https://zenodo.org/record/4743815/files/clotho_captions_development.csv?download=1',
				'hash': '800633304e73d3daed364a2ba6069827',
			},
			'metadata': {
				'fname': 'clotho_metadata_development.csv',
				'url': 'https://zenodo.org/record/4743815/files/clotho_metadata_development.csv?download=1',
				'hash': '5fdc51b4c4f3468ff7d251ea563588c9',
			},
		},
		'eval': {
			'audio_archive': {
				'fname': 'clotho_audio_evaluation.7z',
				'url': 'https://zenodo.org/record/4743815/files/clotho_audio_evaluation.7z?download=1',
				'hash': '4569624ccadf96223f19cb59fe4f849f',
			},
			'captions': {
				'fname': 'clotho_captions_evaluation.csv',
				'url': 'https://zenodo.org/record/4743815/files/clotho_captions_evaluation.csv?download=1',
				'hash': '1b16b9e57cf7bdb7f13a13802aeb57e2',
			},
			'metadata': {
				'fname': 'clotho_metadata_evaluation.csv',
				'url': 'https://zenodo.org/record/4743815/files/clotho_metadata_evaluation.csv?download=1',
				'hash': '13946f054d4e1bf48079813aac61bf77',
			},
		},
		'val': {
			'audio_archive': {
				'fname': 'clotho_audio_validation.7z',
				'url': 'https://zenodo.org/record/4743815/files/clotho_audio_validation.7z?download=1',
				'hash': '0475bfa5793e80f748d32525018ebada',
			},
			'captions': {
				'fname': 'clotho_captions_validation.csv',
				'url': 'https://zenodo.org/record/4743815/files/clotho_captions_validation.csv?download=1',
				'hash': '3109c353138a089c7ba724f27d71595d',
			},
			'metadata': {
				'fname': 'clotho_metadata_validation.csv',
				'url': 'https://zenodo.org/record/4743815/files/clotho_metadata_validation.csv?download=1',
				'hash': 'f69cfacebcd47c4d8d30d968f9865475',
			},
		},
		'test': {
			'audio_archive': {
				'fname': 'clotho_audio_test.7z',
				'url': 'https://zenodo.org/record/3865658/files/clotho_audio_test.7z?download=1',
			},
			'metadata': {
				'fname': 'clotho_metadata_test.csv',
				'url': 'https://zenodo.org/record/3865658/files/clotho_metadata_test.csv?download=1',
			},
		}
	},
	'v2.1': {
		'dev': {
			'audio_archive': {
				'fname': 'clotho_audio_development.7z',
				'url': 'https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1',
				'hash': 'c8b05bc7acdb13895bb3c6a29608667e',
			},
			'captions': {
				'fname': 'clotho_captions_development.csv',
				'url': 'https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1',
				'hash': 'd4090b39ce9f2491908eebf4d5b09bae',
			},
			'metadata': {
				'fname': 'clotho_metadata_development.csv',
				'url': 'https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1',
				'hash': '170d20935ecfdf161ce1bb154118cda5',
			},
		},
		'eval': {
			'audio_archive': {
				'fname': 'clotho_audio_evaluation.7z',
				'url': 'https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1',
				'hash': '4569624ccadf96223f19cb59fe4f849f',
			},
			'captions': {
				'fname': 'clotho_captions_evaluation.csv',
				'url': 'https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1',
				'hash': '1b16b9e57cf7bdb7f13a13802aeb57e2',
			},
			'metadata': {
				'fname': 'clotho_metadata_evaluation.csv',
				'url': 'https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1',
				'hash': '13946f054d4e1bf48079813aac61bf77',
			},
		},
		'val': {
			'audio_archive': {
				'fname': 'clotho_audio_validation.7z',
				'url': 'https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1',
				'hash': '7dba730be08bada48bd15dc4e668df59',
			},
			'captions': {
				'fname': 'clotho_captions_validation.csv',
				'url': 'https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1',
				'hash': '5879e023032b22a2c930aaa0528bead4',
			},
			'metadata': {
				'fname': 'clotho_metadata_validation.csv',
				'url': 'https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1',
				'hash': '2e010427c56b1ce6008b0f03f41048ce',
			},
		},
		'test': {
			'audio_archive': {
				'fname': 'clotho_audio_test.7z',
				'url': 'https://zenodo.org/record/3865658/files/clotho_audio_test.7z?download=1',
				'hash': '9b3fe72560a621641ff4351ba1154349',
			},
			'metadata': {
				'fname': 'clotho_metadata_test.csv',
				'url': 'https://zenodo.org/record/3865658/files/clotho_metadata_test.csv?download=1',
				'hash': '52f8ad01c229a310a0ff8043df480e21',
			},
		},
	},
}

CLOTHO_AUDIO_DNAMES = {
	'dev': 'development',
	'eval': 'evaluation',
	'test': 'test',
	'val': 'validation',
}


class ClothoDict(Clotho):
	def __getitem__(self, index: int) -> dict[str, Any]:
		fname = self.get_audio_fname(index)
		audio = self.get_audio(index)
		captions = self.get_captions(index)
		metadata = self.get_metadata(index)
		keywords = metadata['keywords'].split(';') if 'keywords' in metadata.keys() else []
		return {
			'index': index,
			'fname': fname,
			'audio': audio,
			'captions': captions,
			'keywords': keywords,
		}
