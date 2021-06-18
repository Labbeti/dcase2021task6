
import json
import logging
import os.path as osp
import tqdm

from typing import Callable, Sequence, Sized


# Start of sentence token
TOKEN_SOS = '<sos>'
# End of sentence token
TOKEN_EOS = '<eos>'
# Unknown token
TOKEN_UNKNOWN = '<unk>'
# Ignore index
IGNORE_ID = -1
# Punctuations tokens
PUNCTUATIONS = {'.', ',', '!', '?', "'", '"', ';', ':', '/'}


def load_vocabulary(fpath: str, verbose: bool = True) -> dict[str, int]:
	assert osp.isfile(fpath), f'Cannot load vocabulary from path "{fpath}".'
	with open(fpath, 'r') as file:
		vocabulary = json.load(file)
		
	if verbose:
		logging.info(f'Compute vocabulary from JSON file "{fpath}", found {len(vocabulary)} distinct words.')
	return vocabulary


def save_vocabulary(vocabulary: dict[str, int], fpath: str, verbose: bool = True):
	if verbose:
		logging.info(f'Save vocabulary of {len(vocabulary)} words to file "{fpath}"...')
	with open(fpath, 'w') as file:
		json.dump(vocabulary, file, indent='\t')


def compute_word_freqs(datasets: Sequence, tokenizer: Callable, add_sos_and_eos: bool = True, verbose: bool = True) -> dict[str, int]:
	assert all((isinstance(dataset, Sized) and hasattr(dataset, 'get_captions') for dataset in datasets))

	word_freqs = {}

	n_samples = sum((len(dataset) for dataset in datasets))
	pbar = tqdm.tqdm(total=n_samples) if verbose else None
	
	for dataset in datasets:
		for i in range(len(dataset)):
			captions = dataset.get_captions(i)
			for caption in captions:
				caption_words = tokenizer(caption)
				for word in caption_words:
					if word in word_freqs.keys():
						word_freqs[word] += 1
					else:
						word_freqs[word] = 1

			if pbar is not None:
				pbar.update(1)

	if pbar is not None:
		pbar.close()

	if add_sos_and_eos:
		if TOKEN_SOS not in word_freqs.keys():
			word_freqs[TOKEN_SOS] = n_samples
		if TOKEN_EOS not in word_freqs.keys():
			word_freqs[TOKEN_EOS] = n_samples
	
	if verbose:
		logging.info(f'Compute vocabulary from {len(datasets)} datasets subsets, found {len(word_freqs)} distinct words.')

	return word_freqs
