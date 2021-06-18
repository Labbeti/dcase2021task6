
from torch.nn import Module
from torchtext.data.metrics import bleu_score
from typing import Any


class Bleu(Module):
	def __init__(self, max_n: int = 4):
		""" BiLingual Evaluation Understudy (BLEU) metric.

		Output values are in range [0, 1]. Higher is better.
		Use torchtext as backend.
		Original paper : https://www.aclweb.org/anthology/P02-1040.pdf

		Args:
			max_n (int, optional): . Defaults to 4.
		"""
		super().__init__()
		self.max_n = max_n
		self.weights = [1.0 / max_n for _ in range(max_n)]

	def forward(self, hypothesis: list[list[Any]], references: list[list[list[Any]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Number of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		score = bleu_score(hypothesis, references, self.max_n, self.weights)
		return score
