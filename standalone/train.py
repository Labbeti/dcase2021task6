
import hydra
import logging
import os.path as osp
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from aac.callbacks.custom_ckpt import CustomModelCheckpoint
from aac.callbacks.evaluator import Evaluator
from aac.callbacks.log import LogLRCallback
from aac.datamodules.get import get_datamodule
from aac.expt.get import get_exptmodule
from aac.utils.custom_logger import CustomTensorboardLogger
from aac.utils.misc import reset_seed, count_params
from aac.utils.vocabulary import save_vocabulary


@hydra.main(config_path='../config', config_name='train')
def main_train(cfg: DictConfig):
	reset_seed(cfg.seed)
	if cfg.verbose:
		logging.info(f'Configuration:\n{OmegaConf.to_yaml(cfg):s}')
		logging.info(f'Datetime: {cfg.datetime:s}\n')
	torch.autograd.set_detect_anomaly(bool(cfg.debug))
	
	if cfg.expt.name in ['ListenAttendTell', 'CnnTell']:
		add_sos_and_eos = True
	else:
		raise RuntimeError(f'Unknown expt name "{cfg.expt.name}" for "add_sos_and_eos" value.')
	
	# Build LightningModule with 'cfg.expt' (model and optim) parameters
	exptmodule = get_exptmodule(**cfg.expt)

	# Build LightningDataModule with 'cfg.data' (dataset and dataloader) parameters, 'cfg.audio' (audio spectrogram) parameters
	datamodule = get_datamodule(**cfg.data, audio_params=cfg.audio, add_sos_and_eos=add_sos_and_eos)

	# Build custom logger and callbacks
	callbacks = []

	if cfg.save:
		logger = CustomTensorboardLogger(**cfg.log)
		logger.log_hyperparams(params={
			'datetime': cfg.datetime,
		})
		if hasattr(datamodule, 'hparams'):
			logger.log_hyperparams(params=datamodule.hparams)

		checkpoint = CustomModelCheckpoint(
			dirpath=osp.join(logger.log_dir, 'checkpoints'),
			save_last=True,
			save_top_k=1,
			verbose=cfg.verbose,
			**cfg.ckpt,
		)
		callbacks.append(checkpoint)
	else:
		logger = None
		checkpoint = None

	# Add Evaluator for compute test metrics scores at the end of the training (when trainer.test is called)
	evaluator = Evaluator(cfg.path.java, cfg.verbose, save_to_csv=cfg.save)
	callbacks.append(evaluator)

	log_lr = LogLRCallback()
	callbacks.append(log_lr)

	log_gpu_memory = 'all' if cfg.debug else None
	weights_summary = 'full' if cfg.debug else 'top'

	trainer = Trainer(
		logger=logger,
		callbacks=callbacks,
		log_gpu_memory=log_gpu_memory,
		move_metrics_to_cpu=True,
		terminate_on_nan=False,
		deterministic=True,
		accelerator=cfg.accelerator,
		gpus=cfg.gpus,
		max_epochs=cfg.epochs,
		resume_from_checkpoint=cfg.resume,
		weights_summary=weights_summary,
		checkpoint_callback=cfg.save,
	)

	if cfg.resume is not None:
		if not isinstance(cfg.resume, str) or not osp.isfile(cfg.resume):
			raise RuntimeError(f'Invalid resume checkpoint fpath "{cfg.resume}".')
		# Prepare and attach data for build model
		datamodule.prepare_data()
		trainer.datamodule = datamodule
		exptmodule.trainer = trainer
		exptmodule.prepare_data()

		# Load best model before testing
		checkpoint_data = torch.load(cfg.resume, map_location=exptmodule.device)
		exptmodule.load_state_dict(checkpoint_data['state_dict'])

	trainer.fit(exptmodule, datamodule=datamodule)

	if cfg.test:
		# Load best model before testing
		if checkpoint is not None and osp.isfile(checkpoint.best_model_path):
			if cfg.verbose:
				logging.info(f'Test using best model "{checkpoint.best_model_path}".')
			checkpoint_data = torch.load(checkpoint.best_model_path, map_location=exptmodule.device)
			exptmodule.load_state_dict(checkpoint_data['state_dict'])
		else:
			if cfg.verbose:
				logging.info(f'Cannot find best model, use last weights.')

		trainer.test(exptmodule, datamodule=datamodule)
	
	if logger is not None:
		if hasattr(datamodule, 'vocabulary'):
			vocab_fpath = osp.join(logger.log_dir, 'vocabulary.json')
			save_vocabulary(datamodule.vocabulary, vocab_fpath)
	
		logger.log_hyperparams(params={
			'expt_n_params': count_params(exptmodule, only_trainable=False),
			'expt_n_params_trainable': count_params(exptmodule, only_trainable=True),
		})

		logger.save_and_close()


if __name__ == '__main__':
	main_train()
