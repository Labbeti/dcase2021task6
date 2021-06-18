
import hydra
import language_tool_python
import logging
import nltk
import os.path as osp
import subprocess
import sys

from omegaconf import DictConfig, OmegaConf
from subprocess import CalledProcessError
from torchaudio.datasets.utils import download_url

from aac.datasets.clotho import Clotho
from aac.models.pann.urls import PANN_PRETRAINED_URLS


@hydra.main(config_path='../config', config_name='download')
def main_download(cfg: DictConfig):
	logging.info(f'Configuration:\n{OmegaConf.to_yaml(cfg):s}')

	if cfg.nltk:
		# Download NLTK model for METEOR metric
		nltk.download('wordnet')

	if cfg.spacy:
		# Download spaCy model for Tokenizer
		model = 'en_core_web_sm'
		command = f'{sys.executable} -m spacy download {model}'.split(' ')
		try:
			subprocess.check_output(command)
			logging.info(f'Model "{model}" for spaCy downloaded.')
		except CalledProcessError as err:
			logging.error(f'Cannot download spaCy model "{model}" for tokenizer. (command "{command}" with exitcode={err.returncode})')
	
	if cfg.pann:
		for i, (name, model_info) in enumerate(PANN_PRETRAINED_URLS.items()):
			path = osp.join(cfg.path.pretrained, model_info['fname'])

			if osp.isfile(path):
				logging.info(f'Model "{name}" already downloaded. ({i+1}/{len(PANN_PRETRAINED_URLS)})')
			else:
				logging.info(f'Start downloading pre-trained PANN model "{name}" ({i}/{len(PANN_PRETRAINED_URLS)})...')
				download_url(model_info['url'], osp.dirname(path), osp.basename(path), model_info['hash'], 'md5')
				logging.info(f'Model "{name}" downloaded.')

	# Download a dataset	
	if cfg.data.name == 'Clotho':
		if cfg.data.version == 'v1':
			subsets = ('dev', 'eval', 'test')
		elif cfg.data.version in ['v2', 'v2.1']:
			subsets = ('dev', 'eval', 'test', 'val')
		else:
			raise RuntimeError(f'Unknown Clotho version "{cfg.data.version}". Must be "v1", "v2" or "v2.1".')

		for subset in subsets:
			_ = Clotho(
				root=cfg.data.root,
				subset=subset,
				version=cfg.data.version,
				download=True,
				verbose=2,
			)
	
	elif cfg.data.name == 'none':
		pass

	else:
		raise RuntimeError(f'Unknown dataset "{cfg.dataset}". Must be "Clotho" or "none".')
	

if __name__ == '__main__':
	main_download()
