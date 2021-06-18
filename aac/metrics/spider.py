
from torch.nn import Module

from .cider import Cider
from .spice import Spice


class Spider(Module):
	"""
		Compute Spider score from cider and spice last scores.
		Useful for avoid to compute the slow Cider metric twice.

		Output values are in range [0, 1]. Higher is better.

		>>> 'spider = (cider + spice) / 2'
	"""
	def __init__(self, cider: Cider, spice: Spice, cider_weight: float = 0.5, spice_weight: float = 0.5):
		assert hasattr(cider, 'get_last_score') and callable(cider.get_last_score)
		assert hasattr(spice, 'get_last_score') and callable(spice.get_last_score)

		super().__init__()
		self.cider = cider
		self.spice = spice
		self.cider_weight = cider_weight
		self.spice_weight = spice_weight

		self._last_score = -1.0

	def forward(self, hypothesis: list[list[str]], references: list[list[list[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Number of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		score_cider = self.cider.get_last_score()
		score_spice = self.spice.get_last_score()
		assert score_cider != -1.0 and score_spice != -1.0
		score = self.cider_weight * score_cider + self.spice_weight * score_spice
		self._last_score = score
		return score

	def get_last_score(self) -> float:
		return self._last_score
