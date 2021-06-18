import unicodedata

from functools import cache
from language_tool_python import LanguageTool
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, Optional, Sized


class DatasetWrap(Dataset):
	def __init__(self, dataset: Dataset, recurse_getattr: bool = True) -> None:
		super().__init__()
		self._dataset = dataset
		self._recurse_getattr = recurse_getattr
	
	def __getitem__(self, index: int) -> Any:
		return self._dataset.__getitem__(index)

	def __len__(self) -> int:
		if isinstance(self._dataset, Sized):
			return len(self._dataset)
		else:
			raise NotImplementedError(f'Wrapped dataset "{self._dataset.__class__.__name__}" is not Sized.')

	def __getattribute__(self, name: str) -> Any:
		try:
			return super().__getattribute__(name)
		except AttributeError:
			pass
			
		message = f'Wrapper {self.__class__.__name__} does not have the attribute "{name}".'
		if not self._recurse_getattr:
			raise AttributeError(message)
		else:
			try:
				return self._dataset.__getattribute__(name)
			except AttributeError:
				raise AttributeError(message)


class PostTransformWrap(DatasetWrap):
	def __init__(self, dataset: Dataset, transform: Optional[Callable], index: Optional[int] = None) -> None:
		super().__init__(dataset)
		self._transform = transform
		self._index = index

	def __getitem__(self, index: int) -> tuple:
		item = self._dataset.__getitem__(index)
		if self._transform is None:
			return item
		elif self._index is None:
			return self._transform(item)
		else:
			return tuple((self._transform(sub_item) if i == self._index else sub_item) for i, sub_item in enumerate(item))


class CacheWrap(DatasetWrap):
	def __init__(self, dataset: Dataset) -> None:
		super().__init__(dataset)
	
	@cache
	def __getitem__(self, index: int) -> tuple:
		return super().__getitem__(index)

	@cache
	def __len__(self) -> int:
		return super().__len__()


def post_process_caption(caption: str, tool: Optional[LanguageTool], remove_accents: bool = False) -> str:	
	excluded = ['*', '|']
	output = caption
	for string in excluded:
		output = output.replace(string, ' ')

	if tool is not None:
		output = tool.correct(caption)

	while '  ' in output:
		output = output.replace('  ', ' ')
	output = output.replace(' -', '-')

	if remove_accents:
		output = unicodedata.normalize('NFD', output).encode('ascii', 'ignore').decode()	
	output = output.strip()
	return output
