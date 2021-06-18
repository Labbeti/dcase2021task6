
from torch.nn import Module

from .spice_coco import Spice as SpiceCoco


class Spice(Module):
	def __init__(self, n_threads: int = 4, java_path: str = 'java'):
		"""
			Semantic Propositional Image Caption Evaluation (SPICE).

			Output values are in range [0, 1]. Higher is better.

			Original paper : https://arxiv.org/pdf/1607.08822.pdf
		"""
		super().__init__()
		self._spice_coco = SpiceCoco(n_threads, java_path)
		self._last_score = -1.0

	def forward(self, hypothesis: list[list[str]], references: list[list[list[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Number of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		res = {i: [' '.join(hyp)] for i, hyp in enumerate(hypothesis)}
		gts = {i: [' '.join(ref) for ref in refs] for i, refs in enumerate(references)}

		average_score, _scores = self._spice_coco.compute_score(gts, res)
		average_score = float(average_score)
		self._last_score = average_score
		return average_score

	def get_last_score(self) -> float:
		return self._last_score
