
from rouge_metric import PyRouge
from torch.nn import Module


class RougeL(Module):
	def __init__(self):
		"""
			Recall Oriented Understudy of Gisting Evaluation.
			
			Output values are in range [0, 1]. Higher is better.
			Use 'rouge-metric' package as backend.

			Original paper: https://www.aclweb.org/anthology/W04-1013.pdf
		"""
		super().__init__()
		self.rouge = PyRouge(rouge_l=True)

	def forward(self, hypothesis: list[list[str]], references: list[list[list[str]]]) -> float:
		if len(hypothesis) != len(references):
			raise ValueError(f'Number of hypothesis and references are different ({len(hypothesis)} != {len(references)}).')

		hypothesis_join = [' '.join(hyp) for hyp in hypothesis]
		references_join = [[' '.join(ref) for ref in refs] for refs in references]

		scores = self.rouge.evaluate(hypotheses=hypothesis_join, multi_references=references_join)
		rouge_l_scores = scores['rouge-l']
		# 3 scores = Recall r, Precision p, FScore f
		# {'r': ..., 'p': ..., 'f': ...}
		f_score = rouge_l_scores['f']

		return f_score
