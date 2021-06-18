import torch

from torch import Tensor
from torch.nn import Module
from typing import Any, Union

from aac.transforms.pad import Pad
from aac.utils.vocabulary import IGNORE_ID


class PadCollate(Module):
	""" 
		Pad audio and captions for LAT model. 
		Also sort audios and captions by audio len for RNN. 

		input: list[audio, captions]	
		output: audio batch, audio lens, captions batch, captions lens
	"""
	def __init__(
		self, 
		caption_fill: int = IGNORE_ID, 
		audio_fill: float = 0.0, 
		has_multiple_captions: bool = True, 
		captions_select: Union[str, int] = 'random', 
		sort_batch: bool = True,
	) -> None:
		"""Pad audio and caption into a batch for dataloaders.

		:param caption_fill: Value used for fill captions for pack them into a batch tensor.
		:param audio_fill: Value used for fill audios for pack them into a batch tensor. 
			(default: 0.0)
		:param has_multiple_captions: If True, consider captions as a list of captions instead of only one unique caption per sample. 
			(default: True)
		:param captions_select: Caption selection per sample. Can be one of ('all', 'random') or the index of the caption. 
			This parameter is ignored if has_multiple_captions is False.
			(default: 'random')
		"""
		super().__init__()
		self.caption_fill = caption_fill
		self.audio_fill = audio_fill
		self.has_multiple_captions = has_multiple_captions
		self.captions_select = captions_select
		self._sort_batch = sort_batch

	def forward(self, batch: list[tuple[Tensor, list]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		# Note: list[a,b] -> (list[a], list[b])
		audios_batch, captions_batch = list(zip(*batch))

		# Pad audio into a batch
		audios_batch_pad, audios_lens = _pad_and_collate_audio(audios_batch, self.audio_fill)

		# Pad captions into a batch
		captions_batch_pad, captions_lens = self._pad_captions(captions_batch)

		if self._sort_batch:
			# Sort audio lens for RNN
			indices = torch.argsort(audios_lens, descending=True)
			audios_batch_pad = audios_batch_pad[indices]
			audios_lens = audios_lens[indices]
			captions_batch_pad = captions_batch_pad[indices]
			captions_lens = captions_lens[indices]
		
		assert len(audios_batch_pad) == len(audios_lens) == len(captions_batch_pad) == len(captions_lens)
		return audios_batch_pad, audios_lens, captions_batch_pad, captions_lens
	
	def _pad_captions(self, captions_batch: list[list]) -> tuple[Tensor, Tensor]:
		if self.has_multiple_captions:
			if self.captions_select == 'all':
				# Get all captions
				captions_batch_pad, captions_lens = _pad_and_collate_captions(captions_batch, self.caption_fill)

			elif self.captions_select == 'random':
				# Get one of the captions
				captions_batch_select = []
				for captions in captions_batch:
					idx = torch.randint(0, len(captions), ())
					captions_batch_select.append(captions[idx])
				captions_batch_pad, captions_lens = _pad_and_collate_caption(captions_batch_select, self.caption_fill)

			elif isinstance(self.captions_select, int):
				# Get the caption n°idx
				idx = self.captions_select
				captions_batch_select = [captions[idx] for captions in captions_batch]
				captions_batch_pad, captions_lens = _pad_and_collate_caption(captions_batch_select, self.caption_fill)

			else:
				raise RuntimeError(f'Unknown multiple caption selection "{self.captions_select}". Must be one of "all", "random" or the index of the caption.')

		else:
			captions_batch_pad, captions_lens = _pad_and_collate_caption(captions_batch, self.caption_fill)
		return captions_batch_pad, captions_lens


def _pad_and_collate_audio(audios_list: list[Tensor], fill_value: float) -> tuple[Tensor, Tensor]:
	"""Pad and stack audio element into a tensor and return a tensor of audio lengths.
	"""
	time_dim = -1
	audios_lens = torch.as_tensor([audio.shape[time_dim] for audio in audios_list])

	max_audio_len = int(audios_lens.max().item())
	pad = Pad(max_audio_len, fill_value=fill_value, dim=time_dim)

	audios_batch_pad = torch.stack([
		pad(audio) for audio in audios_list
	])
	return audios_batch_pad, audios_lens


def _pad_and_collate_captions(captions_list: list[list[list[int]]], fill_value: float) -> tuple[Tensor, Tensor]:
	captions_lens = [[len(caption) for caption in captions] for captions in captions_list]
	captions_lens = torch.as_tensor(captions_lens)

	if captions_lens.numel() == 0:
		return torch.as_tensor(captions_list), captions_lens

	max_caption_len = int(captions_lens.max().item())
	pad = Pad(max_caption_len, fill_value=fill_value, dim=-1)

	captions_batch_pad = torch.stack([
		torch.stack([
			pad(torch.as_tensor(caption)) for caption in captions
		]) for captions in captions_list
	])

	return captions_batch_pad, captions_lens


def _pad_and_collate_caption(caption_list: list[list[int]], fill_value: float) -> tuple[Tensor, Tensor]:
	captions_lens = [len(caption) for caption in caption_list]
	captions_lens = torch.as_tensor(captions_lens)

	if captions_lens.numel() == 0:
		return torch.as_tensor(caption_list), captions_lens

	max_caption_len = int(captions_lens.max().item())
	pad = Pad(max_caption_len, fill_value=fill_value, dim=-1)
	captions_batch_pad = torch.stack([
		pad(torch.as_tensor(caption)) for caption in caption_list
	])
	return captions_batch_pad, captions_lens


class PadCollateDict(Module):
	def __init__(
		self, 
		select_captions: Union[str, int] = 'all', 
		enforce_sorted: bool = True,
		audio_fill: float = 0.0, 
		captions_fill: int = -1, 
		keywords_fill: int = -1,
	):
		super().__init__()
		self._select_captions = select_captions
		self._enforce_sorted = enforce_sorted
		self._audio_fill = audio_fill
		self._captions_fill = captions_fill
		self._keywords_fill = keywords_fill

	def forward(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
		self._check_batch(batch)

		# list[dict[K, V]] -> dict[K, list[V]]
		keys = batch[0].keys()
		result: dict[str, Any] = {key: [dic[key] for dic in batch] for key in keys}
		
		# Pad audio --------------------------------------------------------------------
		audio = result['audio']
		audio, audio_lens = self._pad_list(audio, self._audio_fill)
		# (bsize, audio_max_len)
		result['audio'] = audio
		# (bsize, )
		result['audio_lens'] = audio_lens
		
		# Pad keywords --------------------------------------------------------------------
		keywords = result['keywords']
		keywords, keywords_lens = self._pad_list(keywords, self._keywords_fill)
		# (bsize, keywords_max_len)
		result['keywords'] = keywords
		# (bsize, )
		result['keywords_lens'] = keywords_lens

		# Pad captions --------------------------------------------------------------------
		captions = result['captions']

		if self._select_captions == 'all':
			captions_tuples = [self._pad_list(captions_lst, self._captions_fill) for captions_lst in captions]
			captions, captions_lens = list(zip(*captions_tuples))
			# (bsize, n_captions, caption_max_len)
			result['captions'] = torch.stack(captions)
			# (bsize, n_captions)
			result['captions_lens'] = torch.stack(captions_lens)

		elif self._select_captions == 'random' or isinstance(self._select_captions, int):
			if self._select_captions == 'random':
				bsize = len(captions)
				n_captions = len(captions[0])
				index = torch.randint(0, n_captions, size=(bsize,))
			else:
				index = self._select_captions

			captions = [captions_lst[index] for captions_lst in captions]
			captions, captions_lens = self._pad_list(captions, self._captions_fill)
			# (bsize, caption_max_len)
			result['captions'] = captions
			# (bsize, )
			result['captions_lens'] = captions_lens
		
		else:
			raise RuntimeError(f'Unknown multiple caption selection "{self.captions_select}". Must be one of "all", "random" or the index of the caption.')

		# Sort values in decreasing order of audio --------------------------------------------------------------------
		if self._enforce_sorted:
			indexes = torch.argsort(audio_lens)
			for key, values in result.items():
				if isinstance(values, Tensor):
					result[key] = values[indexes]
				else:
					result[key] = [values[idx] for idx in indexes]

		return result

	def _pad_list(self, lst: list[Union[Tensor, list[int]]], fill_value: Union[int, float]) -> tuple[Tensor, Tensor]:
		lens = torch.as_tensor([len(elt) for elt in lst])
		max_audio_len = int(lens.max().item())
		pad = Pad(max_audio_len, fill_value=fill_value, dim=-1)

		packed_lst = torch.stack([
			pad(torch.as_tensor(elt)) for elt in lst
		])
		return packed_lst, lens

	def _check_batch(self, batch: list[dict[str, Any]]):
		if len(batch) == 0:
			raise RuntimeError(f'Empty batch found.')
		
		expected_keys = set(batch[0].keys())
		for dic in batch[1:]:
			if expected_keys != set(dic.keys()):
				raise RuntimeError(f'Found a dict with different keys in batch. ({expected_keys} != {set(dic.keys())})')
