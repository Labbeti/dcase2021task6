
import re
import spacy
import torch

from torch import Tensor
from torch.nn import Module, Sequential
from torchtext.data.utils import get_tokenizer
from typing import Callable, Iterable, Optional, Union

from aac.utils.vocabulary import TOKEN_SOS, TOKEN_EOS, TOKEN_UNKNOWN


class Tokenizer(Module):
	""" 
		Clean sentence, tokenize to words and add SOS and EOS tokens.
	"""
	def __init__(
		self,
		keep_case: bool = False, 
		remove_punctuation: bool = True, 
		add_sos_and_eos: bool = True,
		backend: str = 'spacy',
	) -> None:
		super().__init__()
		self._keep_case = keep_case
		self._remove_punctuation = remove_punctuation
		self._add_sos_and_eos = add_sos_and_eos
		self._backend = backend

		if backend == 'torchtext':
			self._tokenizer = get_tokenizer('basic_english')
		elif backend == 'spacy':
			nlp = spacy.load('en_core_web_sm')
			self._tokenizer = nlp.tokenizer
		else:
			raise RuntimeError(f'Unknown backend "{backend}".')
	
	def forward(self, input_: Union[str, Iterable]) -> Union[str, list]:
		if isinstance(input_, str):
			sentence = input_
			sentence = self.clean_sentence(sentence)
			words = self.tokenize(sentence)
			words = self.add_sos_and_eos(words)
			return words
		elif isinstance(input_, Iterable):
			return [self.forward(elt) for elt in input_]
		else:
			raise RuntimeError(f'Unknown sentence type "{type(input_)}".')

	def clean_sentence(self, sentence: str) -> str:
		"""
			Based on function 'https://github.com/audio-captioning/clotho-dataset/blob/8a1981da4722a60cb3ed6424e3613edbaa124d82/tools/captions_functions.py#L47'

			Add 5 punctuation tokens : “”’(){}[]*× for AudioCaps
		"""
		if not self._keep_case:
			sentence = sentence.lower()

		# Remove any forgotten space before punctuation and double space.
		sentence = re.sub(r'\s([,.!?;:"“”’\(\)\{\}\[\]\*\×)](?:\s|$))', r'\1', sentence)

		# Remove punctuation
		if self._remove_punctuation:
			sentence = re.sub(r'[,.!?;:\"“”’\(\)\{\}\[\]\*\×]', ' ', sentence)

		# Remove all double spaces
		while '  ' in sentence:
			sentence = sentence.replace('  ', ' ')

		sentence = sentence.strip()
		return sentence
	
	def tokenize(self, caption: str) -> list[str]:
		outputs = self._tokenizer(caption)
		if self._backend == 'spacy':
			# Spacy returns a list of spacy.tokens.token.Token object, so get the actual string with 'text' property
			outputs = [token.text for token in outputs]
		return outputs

	def add_sos_and_eos(self, words: list[str]) -> list[str]:
		if self._add_sos_and_eos:
			return [TOKEN_SOS] + words + [TOKEN_EOS]
		else:
			return words


class WordsToIdx(Module):
	""" Map word string to index. """
	def __init__(self, word_to_idx: dict[str, int]):
		super().__init__()
		self.word_to_idx = word_to_idx

	def forward(self, input_: Union[str, list]) -> Union[int, list]:
		if isinstance(input_, str):
			return self.word_to_idx[input_]
		elif isinstance(input_, (list, tuple)):
			return [self.forward(elt) for elt in input_]
		else:
			raise RuntimeError(f'Unknown sentence type "{type(input_)}".')


class IdxToWords(Module):
	""" Map word index to string. """
	def __init__(self, idx_to_words: dict[int, str], raise_on_unknown: bool = False):
		super().__init__()
		self.idx_to_words = idx_to_words
		self.raise_on_unknown = raise_on_unknown

	def forward(self, input_: Union[Tensor, int, list]) -> Union[str, list]:
		if isinstance(input_, Tensor):
			input_ = input_.tolist()

		if isinstance(input_, int):
			if input_ in self.idx_to_words.keys():
				return self.idx_to_words[input_]
			elif self.raise_on_unknown:
				raise KeyError(f'Unknown index "{input_}" for IdxToWords.')
			else:
				return TOKEN_UNKNOWN
		elif isinstance(input_, (list, tuple)):
			return [self.forward(elt) for elt in input_]
		else:
			raise RuntimeError(f'Unknown sentence type "{type(input_)}".')


class CutSentencesAtTokens(Module):
	def __init__(self, *tokens: str):
		super().__init__()
		self.tokens = list(tokens)

	def forward(self, sentences: list[list[str]]) -> list[list[str]]:
		assert isinstance(sentences, list)

		result = []
		for sentence in sentences:
			idx = None
			for i, word in enumerate(sentence):
				if word in self.tokens:
					idx = i
					break
			result.append(sentence[:idx])
		return result


class ExcludeWords(Module):
	"""
		Example :

		>>> ew = ExcludeWords('!', ',')
		>>> ew(['a', 'thing', ',', 'here', '!'])
		... ['a', 'thing', 'here']
	"""
	def __init__(self, *excluded: Union[str, Iterable[str]]):
		super().__init__()
		if len(excluded) == 1 and isinstance(excluded[0], Iterable):
			self.excluded = set(excluded[0])
		else:
			self.excluded = set(excluded)

	def forward(self, sentences: list) -> list:
		if isinstance(sentences, list) and len(sentences) > 0 and isinstance(sentences[0], str):
			return [word for word in sentences if word not in self.excluded]
		elif isinstance(sentences, (list, tuple)):
			return [self.forward(elt) for elt in sentences]
		else:
			raise RuntimeError(f'Unknown sentence type "{type(sentences)}".')


class ExcludeWordsBelowFreq(Module):
	def __init__(self, word_freqs: dict[str, int], min_freq: int = 0) -> None:
		super().__init__()
		self.word_freqs = word_freqs
		self.min_freq = min_freq

	def forward(self, captions: list[list[str]]) -> None:
		return [
			[word for word in caption if self.word_freqs[word] >= self.min_freq]
			for caption in captions
		]


class SelectCaptionList(Module):
	def __init__(self, mode: Union[int, str] = 'random'):
		super().__init__()
		self.mode = mode
	
	def forward(self, captions: list[list[str]]) -> list:
		if self.mode == 'random':
			index = int(torch.randint(0, len(captions), ()).item())
		elif self.mode == 'all':
			return captions
		elif isinstance(self.mode, int) and 0 <= self.mode < len(captions):
			index = self.mode
		else:
			raise RuntimeError(f'Invalid SelectCaption mode "{self.mode}". Must be one of ("random", "all" or positive index).')
		
		return captions[index]


class SelectCaptionTensor(Module):
	def __init__(self, mode: Union[int, str] = 'random', dim: int = 0):
		super().__init__()
		self.mode = mode
		self.dim = dim
	
	def forward(self, captions: Tensor) -> Tensor:
		if self.mode == 'random':
			index = int(torch.randint(0, len(captions), ()).item())
		elif self.mode == 'all':
			return captions
		elif isinstance(self.mode, int) and 0 <= self.mode < len(captions):
			index = self.mode
		else:
			raise RuntimeError(f'Invalid SelectCaption mode "{self.mode}". Must be one of ("random", "all" or positive index).')
		
		# Example if captions has 3 dims and self.dim=1 : captions = captions[:, idx, :]
		caption = torch.select(captions, self.dim, index).contiguous()
		return caption


class WordFunc(Module):
	def __init__(self, func: Callable[[str], str]):
		super().__init__()
		self._func = func
	
	def forward(self, input_: Union[str, list]) -> Union[str, list]:
		if isinstance(input_, str):
			return self.process_func(input_)
		elif isinstance(input_, Iterable):
			return [self.forward(subinput) for subinput in input_]
		else:
			raise RuntimeError(f'Invalid input type "{type(input_)}".')
	
	def process_func(self, input_: str) -> str:
		return self._func(input_)
