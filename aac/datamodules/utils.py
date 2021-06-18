
import logging

from torch.utils.data.dataset import Dataset, Subset
from typing import Callable, Sized


def default_filter(sentence: str) -> bool:
	# TODO : rem ?
	invalid_string = [str(num) for num in range(10)]
	invalid_string += ['|']
	return all((string not in sentence for string in invalid_string))


def filter_dataset(dataset: Dataset, filter_caption: Callable[[str], bool], verbose: bool = False) -> Dataset:
	# TODO : rem ?
	assert isinstance(dataset, Sized) and hasattr(dataset, 'get_captions')
	indexes = []
	for i in range(len(dataset)):
		captions = dataset.get_captions(i)
		pass_filter = all((filter_caption(caption) for caption in captions))
		if pass_filter:
			indexes.append(i)

	if verbose:
		logging.info(f'Filter dataset "{dataset.__class__.__name__}" : {len(indexes)}/{len(dataset)} items are kept.')
	return Subset(dataset, indexes)
