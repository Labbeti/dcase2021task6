
import csv
import logging
import os.path as osp

from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from aac.metrics import (
	Bleu, 
	Meteor, 
	RougeL, 
	Cider, 
	Spice, 
	Spider,
)


class Evaluator(Callback):
	def __init__(self, java_path: str = 'java', verbose: bool = True, save_to_csv: bool = True) -> None:
		self._verbose = verbose
		self._save_to_csv = save_to_csv
		self._outputs = {}

		cider = Cider()
		spice = Spice(java_path=java_path)
		self._metrics = {
			'bleu1': Bleu(1),
			'bleu2': Bleu(2),
			'bleu3': Bleu(3),
			'bleu4': Bleu(4),
			'meteor': Meteor(),
			'rouge_l': RougeL(),
			'cider': cider,
			'spice': spice,
			'spider': Spider(cider, spice),
		}
		self._prefix = 'metrics_'

	def on_test_batch_end(self, trainer, pl_module, outputs: tuple[list, list], batch, batch_idx, dataloader_idx: int) -> None:
		if dataloader_idx not in self._outputs.keys():
			self._outputs[dataloader_idx] = []
		self._outputs[dataloader_idx].append(outputs)

	def on_test_epoch_start(self, trainer, pl_module) -> None:
		self._outputs = {}

	def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
		for dataloader_idx, pl_module_outputs in self._outputs.items():
			# Note : since each output is a batch of prediction (or captions), we remove this intermediate index
			pred_all, captions_all = [], []
			for pred, captions in pl_module_outputs:
				pred_all += pred
				captions_all += captions

			assert len(pred_all) == len(captions_all), f'Number of pred != Number of captions ({len(pred_all)} != {len(captions_all)}).'

			test_dataset = pl_module.trainer.datamodule.test_datasets[dataloader_idx]
			subset = test_dataset.subset if hasattr(test_dataset, 'subset') and isinstance(test_dataset.subset, str) else f'loader{dataloader_idx}'

			self.log_scores(pl_module, pred_all, captions_all, dataloader_idx, subset)
			if pl_module.logger is not None:
				dpath = pl_module.logger.experiment.log_dir
				self.save_predictions_to_csv(dpath, pred_all, captions_all, test_dataset, subset)

	def save_predictions_to_csv(self, dpath: str, pred_all: list, captions_all: list, dataset, subset: str) -> None:
		if not self._save_to_csv:
			return None

		if not osp.isdir(dpath):
			raise RuntimeError(f'Cannot save predictions in csv file. (logdir "{dpath}" is not a dir).')
		if len(pred_all) != len(dataset):
			raise RuntimeError(f'Number of predictions != Length of dataset "{dataset.__class__.__name__}", subset "{subset}" ({len(pred_all)} != {len(dataset)}).')
		if len(captions_all) != len(dataset):
			raise RuntimeError(f'Number of captions != Length of dataset "{dataset.__class__.__name__}", subset "{subset}" ({len(pred_all)} != {len(dataset)}).')
		if not hasattr(dataset, 'get_audio_fpath'):
			raise RuntimeError('Dataset does not have a "get_audio_fpath" method.')

		fpath_csv = osp.join(dpath, f'results_{subset}.csv')
		with open(fpath_csv, 'w') as file:
			writer = csv.DictWriter(file, fieldnames=['file_name', 'caption_predicted'])
			writer.writeheader()

			for i, pred in enumerate(pred_all):
				fpath = dataset.get_audio_fpath(i)
				fname = osp.basename(fpath)
				writer.writerow({
					'file_name': fname,
					'caption_predicted': ' '.join(pred),
				})
		
		max_captions_per_sample = max((len(captions) for captions in captions_all))

		fpath_csv = osp.join(dpath, f'results_full_{subset}.csv')
		with open(fpath_csv, 'w') as file:
			writer = csv.DictWriter(file, fieldnames=['index', 'file_name', 'caption_predicted'] + [f'caption_{j+1}' for j in range(max_captions_per_sample)])
			writer.writeheader()

			for i, (pred, captions) in enumerate(zip(pred_all, captions_all)):
				fpath = dataset.get_audio_fpath(i)
				fname = osp.basename(fpath)
				row = {
					'index': i,
					'file_name': fname,
					'caption_predicted': ' '.join(pred),
				}
				for j, caption in enumerate(captions):
					row[f'caption_{j+1}'] = ' '.join(caption)
				for j in range(len(captions), max_captions_per_sample):
					row[f'caption_{j+1}'] = ''
				writer.writerow(row)
	
	def log_scores(self, pl_module: LightningModule, pred_flat: list, captions_flat: list, dataloader_idx: int, subset: str) -> None:
		# Compute metrics only if captions are not empty
		metric_inputs = [(pred, caption) for pred, caption in zip(pred_flat, captions_flat) if caption != [] and caption != [[]] and caption is not None]
		# list[a, b] -> list[a], list[b]
		metric_inputs = list(zip(*metric_inputs))

		if len(metric_inputs) > 0 and len(metric_inputs[0]) > 0 and len(metric_inputs[1]) > 0:
			if self._verbose:
				logging.info(f'Start to compute metrics... ({", ".join(self._metrics.keys())}) for dataloader {dataloader_idx} and subset "{subset}".')
			# Call test metrics
			scores = {name: metric(*metric_inputs) for name, metric in self._metrics.items()}
			scores = {f'{self._prefix}{subset}/{name}': score for name, score in scores.items()}

			if self._verbose:
				logging.info(f'Saving scores for subset "{subset}" : \n{OmegaConf.to_yaml(scores)}')
				
			pl_module.log_dict(scores, on_epoch=True, on_step=False, logger=False)
			if pl_module.logger is not None:
				pl_module.logger.log_hyperparams(params={}, metrics=scores)
