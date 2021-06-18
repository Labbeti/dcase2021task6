
from torch.nn import Module

from .cider_coco import Cider as CiderCoco


class Cider(Module):
	def __init__(self, ngrams_max: int = 4, sigma: float = 6.0):
		"""
			Consensus-based Image Description Evaluation (CIDEr).

			Output values are in range [0, 1]. Higher is better.
			Original paper : https://arxiv.org/pdf/1411.5726.pdf

			:param ngrams_max: Maximum number of ngrams used. (default: 4)
		"""
		super().__init__()
		self._cider_coco = CiderCoco(n=ngrams_max, sigma=sigma)
		self._last_score = -1.0

	def forward(self, hypothesis: list[list[str]], references: list[list[list[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Number of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		res = {i: [' '.join(hyp)] for i, hyp in enumerate(hypothesis)}
		gts = {i: [' '.join(ref) for ref in refs] for i, refs in enumerate(references)}

		average_score, _scores = self._cider_coco.compute_score(gts, res)
		average_score = float(average_score)
		self._last_score = average_score
		return average_score

	def get_last_score(self) -> float:
		return self._last_score
