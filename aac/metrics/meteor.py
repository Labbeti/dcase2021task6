
import nltk
import torch

from nltk.translate.meteor_score import meteor_score
from torch.nn import Module


class Meteor(Module):
	def __init__(self, alpha: float = 0.9, gamma: float = 0.5, beta: float = 3.0):
		"""
			Metric for Evaluation of Translation with Explicit ORdering (METEOR)

			Output values are in range [0, 1]. Higher is better.
			Use 'nltk' package as backend.

			:param alpha: Parameter for controlling the weights of precision and recall. (default: 0.9)
			:param gamma: The coefficient used in the brevity penalty function. (default: 0.5)
			:param beta: The power used in the brevity penalty function. (default: 3.0)
		"""
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.beta = beta

		# Download nltk wordnet for METEOR metric
		nltk.download('wordnet')

	def forward(self, hypothesis: list[list[str]], references: list[list[list[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Number of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		batch_scores = []
		for hyp, refs in zip(hypothesis, references):
			hyp = ' '.join(hyp)
			refs = [' '.join(ref) for ref in refs]
			score = meteor_score(hypothesis=hyp, references=refs, alpha=self.alpha, beta=self.beta, gamma=self.gamma)
			batch_scores.append(score)

		score = torch.as_tensor(batch_scores)
		score = torch.mean(score)
		score = float(score.item())
		return score
