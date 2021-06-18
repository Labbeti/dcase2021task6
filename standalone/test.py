
import glob
import hydra
import logging
import os.path as osp
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from aac.callbacks.evaluator import Evaluator
from aac.datamodules.get import get_datamodule
from aac.expt.get import get_exptmodule
from aac.utils.custom_logger import CustomTensorboardLogger
from aac.utils.misc import reset_seed, count_params
from aac.utils.vocabulary import save_vocabulary


@hydra.main(config_path='../config', config_name='test')
def main_test(cfg: DictConfig):
	reset_seed(cfg.seed)
	if cfg.verbose:
		logging.info(f'Configuration:\n{OmegaConf.to_yaml(cfg):s}')
		logging.info(f'Datetime: {cfg.datetime:s}\n')
	torch.autograd.set_detect_anomaly(bool(cfg.debug))

	if cfg.resume is None:
		raise RuntimeError(f'Please specify a valid resume filepath with option "resume=PATH_TO_CHECKPOINT.ckpt" (resume="{cfg.resume}").')

	if osp.isfile(cfg.resume):
		resume_fpath = cfg.resume
	else:
		# Use glob for search a checkpoint file with a pattern in "cfg.resume"
		matches = glob.glob(cfg.resume, recursive=True)
		if len(matches) == 0:
			raise RuntimeError(f'Cannot find any match for checkpoint pattern." (resume="{cfg.resume}").')
		elif len(matches) > 1:
			raise RuntimeError(f'Found {len(matches)} matches for checkpoint pattern." (resume="{cfg.resume}", matches="{matches}").')
		else:
			resume_fpath = matches[0]
			if not osp.isfile(resume_fpath):
				raise RuntimeError(f'Match path is not a file. (resume="{cfg.resume}", match="{resume_fpath}")')
	
	if cfg.verbose:
		logging.info(f'Checkpoint filepath : "{resume_fpath}"')

	if cfg.expt.name in ['ListenAttendTell', 'CnnTell']:
		add_sos_and_eos = True
	elif cfg.expt.name in ['SpeechTransformer', 'CplxSpeechTransformer']:
		add_sos_and_eos = False
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
	else:
		logger = None
	
	evaluator = Evaluator(cfg.path.java, cfg.verbose, save_to_csv=True)
	callbacks.append(evaluator)

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
		weights_summary=weights_summary,
		checkpoint_callback=cfg.save,
	)

	# Prepare and attach data for build model
	datamodule.prepare_data()
	trainer.datamodule = datamodule
	exptmodule.trainer = trainer
	exptmodule.prepare_data()

	# Load best model before testing
	checkpoint_data = torch.load(resume_fpath, map_location=exptmodule.device)
	exptmodule.load_state_dict(checkpoint_data['state_dict'], strict=True)

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
	main_test()
