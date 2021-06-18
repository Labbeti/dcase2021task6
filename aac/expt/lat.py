
from aac.models.pann.wavegram import Wavegram
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential
from torch.optim import Optimizer
from typing import Optional, OrderedDict

from aac.models.listen_attend_tell import BeamSeq2Seq, masked_ce_loss, Seq2Seq
from aac.transforms.augments.mixup import Mixup
from aac.transforms.captions import IdxToWords, CutSentencesAtTokens, ExcludeWords
from aac.utils.optimizers import get_optimizer
from aac.utils.schedulers import get_scheduler
from aac.utils.vocabulary import IGNORE_ID, TOKEN_SOS, TOKEN_EOS, TOKEN_UNKNOWN


class LMListenAttendTell(LightningModule):
	def __init__(
		self,
		optim_name: str = 'Adam',
		lr: float = 5e-4,
		weight_decay: float = 1e-6,
		betas: tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-8,
		sched_name: str = 'cos_decay',
		verbose: bool = False,
		use_cutoutspec: bool = False,
		input_dim: int = 64,
		use_spec_augment: bool = False,
		encoder_hidden_dim: int = 128,
		embedding_dim: int = 128,
		decoder_hidden_size_1: int = 128,
		decoder_hidden_size_2: int = 64,
		query_size: int = 64,
		value_size: int = 64,
		key_size: int = 64,
		is_attended: bool = True,
		teacher_forcing_ratio: float = 0.98,
		pBLSTM_time_reductions: list[int] = [2, 2],
		use_gumbel_noise: bool = False,
		beam_size: int = 10,
		use_wavegram: bool = False,
		fpath_wavegram: Optional[str] = None,
		trainable_wavegram: bool = False,
		max_output_len: int = 30,
		use_mixup: bool = False,
		beam_alpha: float = 1.2,
	) -> None:
		super().__init__()
		# Optim hparams
		self.optim_name = optim_name
		self.lr = lr
		self.weight_decay = weight_decay
		self.betas = betas
		self.eps = eps
		self.sched_name = sched_name
		self.verbose = verbose
		self.use_cutoutspec = use_cutoutspec
		# Model hparams
		self.input_dim = input_dim
		self.use_spec_augment = use_spec_augment
		self.encoder_hidden_dim = encoder_hidden_dim
		self.embedding_dim = embedding_dim
		self.decoder_hidden_size_1 = decoder_hidden_size_1
		self.decoder_hidden_size_2 = decoder_hidden_size_2
		self.query_size = query_size
		self.value_size = value_size
		self.key_size = key_size
		self.is_attended = is_attended
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.pBLSTM_time_reductions = pBLSTM_time_reductions
		self.use_gumbel_noise = use_gumbel_noise
		self.beam_size = beam_size
		self.max_output_len = max_output_len
		self.use_mixup = use_mixup
		self.beam_alpha = beam_alpha

		if use_wavegram:
			self.wavegram = Wavegram(fpath_wavegram, trainable_wavegram)
		else:
			self.wavegram = None

		if use_mixup:
			self.mixup = Mixup(alpha=0.4, asymmetric=True)
		else:
			self.mixup = None

		# Metrics
		self.train_metrics = {}
		self.val_metrics = {}

		# Other attributes
		self.criterion = masked_ce_loss
		self.prepared = False
		self.idx_to_word = None
		self.word_to_idx = None
		self.model = None
		self.pred_to_words = None

		self.save_hyperparameters()

	def prepare_data(self) -> None:
		if not self.prepared:
			self.prepared = True
			
			datamodule = self.trainer.datamodule
			self.idx_to_word = datamodule.idx_to_word
			self.word_to_idx = datamodule.word_to_idx

			self.word_to_model_idx = self.word_to_idx
			self.model_idx_to_word = self.idx_to_word

			self.model = Seq2Seq(
				input_dim=self.input_dim,
				vocab_size=len(self.word_to_model_idx),
				word2index=self.word_to_model_idx,
				use_spec_augment=self.use_spec_augment,
				encoder_hidden_dim=self.encoder_hidden_dim,
				embedding_dim=self.embedding_dim,
				decoder_hidden_size_1=self.decoder_hidden_size_1,
				decoder_hidden_size_2=self.decoder_hidden_size_2,
				query_size=self.query_size,
				value_size=self.value_size,
				key_size=self.key_size,
				pBLSTM_time_reductions=self.pBLSTM_time_reductions,
				teacher_forcing_ratio=self.teacher_forcing_ratio,
				isAttended=self.is_attended,
				max_output_len=self.max_output_len,
			)

			# Init transforms
			self.pred_to_words = Sequential(
				IdxToWords(self.idx_to_word),
				CutSentencesAtTokens(TOKEN_EOS),
				ExcludeWords(TOKEN_SOS, TOKEN_EOS, TOKEN_UNKNOWN),
			)
			self.captions_to_words = self.pred_to_words

	def setup(self, stage: Optional[str] = None):
		if stage in ['fit', None]:
			dataloader = self.trainer.datamodule.train_dataloader()
			audios, audios_lens, _, _ = next(iter(dataloader))
			self.example_input_array = (audios, audios_lens)

		if stage in ['test', None]:
			if self.beam_size > 1:
				beam_model = BeamSeq2Seq(
					input_dim=self.input_dim,
					vocab_size=len(self.word_to_model_idx),
					encoder_hidden_dim=self.encoder_hidden_dim,
					use_spec_augment=False,
					embedding_dim=self.embedding_dim,
					decoder_hidden_size_1=self.decoder_hidden_size_1,
					decoder_hidden_size_2=self.decoder_hidden_size_2,
					query_size=self.query_size,
					value_size=self.value_size,
					key_size=self.key_size,
					isAttended=True,
					pBLSTM_time_reductions=self.pBLSTM_time_reductions,
					teacher_forcing_ratio=self.teacher_forcing_ratio,
					beam_size=self.beam_size,
					use_lm_bigram=False,
					use_lm_trigram=False,
					lm_weight=0.0,
					word2index=self.word_to_idx,
					index2word=self.idx_to_word,
					vocab=list(self.word_to_idx.keys()),
					return_attention_masks=False,
					max_output_len=self.max_output_len,
					beam_alpha=self.beam_alpha,
				)
				beam_model.load_state_dict(OrderedDict(self.model.cpu().state_dict()))
				self.model = beam_model

	def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
		"""
			Shapes :
				xs: (time, bsize, freq),
				xs_lens: (bsize,),
				ys: (bsize, max_caption_len),
				ys_lens: (bsize,)
		"""
		audios, audios_lens, captions, captions_lens = batch
		
		if self.wavegram is not None:
			audios = self.wavegram(audios)
			audios_lens = torch.full((audios.shape[0],), fill_value=audios.shape[-1], dtype=torch.long)
		
		if self.mixup is not None:
			audios, _ = self.mixup(audios, captions)

		captions[captions == IGNORE_ID] = self.word_to_idx[TOKEN_EOS]
		# Permute (bsize, freq, time) -> (time, bsize, freq)
		audios = audios.permute(2, 0, 1)

		logits = self.model(audios, audios_lens.cpu(), captions, isTrain=True, use_gumbel_noise=self.use_gumbel_noise)
		loss = self.criterion(logits, captions[:, 1:].contiguous(), captions_lens - 1)

		with torch.no_grad():
			self.log('train/loss', loss.cpu(), on_epoch=True, on_step=False)

			if len(self.train_metrics) > 0:
				pred = torch.argmax(logits, dim=-1)
				pred = self.pred_to_words(pred)
				captions = self.captions_to_words(captions)

				scores = {name: metric(pred, captions) for name, metric in self.train_metrics.items()}
				self.log_dict(scores, on_epoch=True, on_step=False)

		return loss

	def validation_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
		audios, audios_lens, captions, captions_lens = batch
		
		if self.wavegram is not None:
			audios = self.wavegram(audios)
			audios_lens = torch.full((audios.shape[0],), fill_value=audios.shape[-1], dtype=torch.long)

		captions[captions == IGNORE_ID] = self.word_to_idx[TOKEN_EOS]
		# Permute (bsize, freq, time) -> (time, bsize, freq)
		audios = audios.permute(2, 0, 1)

		logits = self.model(audios, audios_lens.cpu(), captions, isTrain=True, use_gumbel_noise=False)
		loss = self.criterion(logits, captions[:, 1:].contiguous(), captions_lens - 1)

		self.log('val/loss', loss.cpu(), on_epoch=True, on_step=False, prog_bar=True)

		if len(self.val_metrics) > 0:
			pred = torch.argmax(logits, dim=-1)
			pred = self.pred_to_words(pred)
			captions = self.captions_to_words(captions)

			scores = {name: metric(pred, captions) for name, metric in self.val_metrics.items()}
			self.log_dict(scores, on_epoch=True, on_step=False)

	def test_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int) -> tuple[list, list]:
		audios, audios_lens, captions, _ = batch
		captions[captions == IGNORE_ID] = self.word_to_idx[TOKEN_EOS]

		pred = self(audios, audios_lens)

		pred = self.pred_to_words(pred)
		captions = self.captions_to_words(captions)
		return pred, captions

	def forward(self, audios: Tensor, audios_lens: Tensor):
		"""Predict the captions with spectrogram of audio.

		Args:
			audios (Tensor): Spectrograms of shape (bsize, freq, time)
			audios_lens (Tensor): Audios lens of shape (bsize,)

		Returns:
			Tensor: The predictions of each audio sample as a tensor of word indexes.
		"""
		if self.wavegram is not None:
			audios = self.wavegram(audios)
			audios_lens = torch.full((audios.shape[0],), fill_value=audios.shape[-1], dtype=torch.long)
		
		if isinstance(self.model, Seq2Seq):
			# Permute (bsize, freq, time) -> (time, bsize, freq)
			audios = audios.permute(2, 0, 1)
			logits = self.model(audios, audios_lens.cpu(), text_input=None, isTrain=False, use_gumbel_noise=False)
			preds = torch.argmax(logits, dim=-1)

		else: # BeamSeq2Seq
			preds = []
			for audio, audio_len in zip(audios, audios_lens):
				audio = audio.unsqueeze(dim=0)
				audio_len = audio_len.unsqueeze(dim=0)
				# Permute (1, freq, time) -> (time, 1, freq)
				audio = audio.permute(2, 0, 1)
				
				pred = self.model(audio, audio_len.cpu(), text_input=None, isTrain=False, use_gumbel_noise=False)
				preds.append(pred)
				
		return preds

	def configure_optimizers(self) -> tuple[list[Optimizer], list]:
		optimizer = get_optimizer(
			self.optim_name, 
			self.model.parameters(), 
			lr=self.lr,
			weight_decay=self.weight_decay,
			betas=self.betas,
			eps=self.eps,
		)
		scheduler = get_scheduler(
			self.sched_name, 
			optimizer, 
			epochs=self.trainer.max_epochs,
		)
		return [optimizer], ([scheduler] if scheduler is not None else [])
