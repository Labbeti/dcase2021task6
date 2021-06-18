
import torch

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential
from typing import Optional

from aac.models.cnnt import CnnTell
from aac.models.listen_attend_tell import masked_ce_loss
from aac.transforms.captions import IdxToWords, CutSentencesAtTokens, ExcludeWords
from aac.utils.optimizers import get_optimizer
from aac.utils.schedulers import get_scheduler
from aac.utils.vocabulary import TOKEN_SOS, TOKEN_EOS, TOKEN_UNKNOWN, IGNORE_ID


class LMCnnTell(LightningModule):
	def __init__(
		self, 
		optim_name: str = 'Adam',
		lr: float = 5e-4,
		weight_decay: float = 1e-6,
		betas: tuple[float, float] = (0.9, 0.999),
		eps: float = 1e-8,
		sched_name: str = 'cos_decay',
		verbose: bool = False,
		dpath_pretrained: Optional[str] = None,
		embedding_dim: int = 128,
		decoder_hidden_size_1: int = 128,
		decoder_hidden_size_2: int = 64,
		query_size: int = 64,
		value_size: int = 64,
		key_size: int = 64,
		is_attended: bool = True,
		teacher_forcing_ratio: float = 0.98,
		beam_size: int = 10,
		max_output_len: int = 30,
		freeze_cnn14: bool = False,
		use_cutout: bool = False,
		use_mixup: bool = False,
		use_specaugm: bool = False,
		beam_alpha: float = 1.2,
	) -> None:
		super().__init__()
		self.optim_name = optim_name
		self.lr = lr
		self.weight_decay = weight_decay
		self.betas = betas
		self.eps = eps
		self.sched_name = sched_name
		self.verbose = verbose

		self.dpath_pretrained = dpath_pretrained
		self.embedding_dim = embedding_dim
		self.decoder_hidden_size_1 = decoder_hidden_size_1
		self.decoder_hidden_size_2 = decoder_hidden_size_2
		self.query_size = query_size
		self.value_size = value_size
		self.key_size = key_size
		self.is_attended = is_attended
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.beam_size = beam_size
		self.max_output_len = max_output_len
		self.freeze_cnn14 = freeze_cnn14
		self.use_cutout = use_cutout
		self.use_mixup = use_mixup
		self.use_specaugm = use_specaugm
		self.beam_alpha = beam_alpha

		self.criterion = masked_ce_loss
		self.prepared = False
		self.word_to_idx = None
		self.idx_to_word = None
		self.model = None

		self.save_hyperparameters()

	def prepare_data(self) -> None:
		if not self.prepared:
			self.prepared = True

			datamodule = self.trainer.datamodule
			self.word_to_idx = datamodule.word_to_idx
			self.idx_to_word = datamodule.idx_to_word

			self.model = CnnTell(
				dpath_pretrained=self.dpath_pretrained,
				word_to_idx=self.word_to_idx,
				n_classes=len(self.word_to_idx),
				embedding_dim=self.embedding_dim,
				decoder_hidden_size_1=self.decoder_hidden_size_1,
				decoder_hidden_size_2=self.decoder_hidden_size_2,
				query_size=self.query_size,
				value_size=self.value_size,
				key_size=self.key_size,
				is_attended=self.is_attended,
				beam_size=self.beam_size,
				teacher_forcing_ratio=self.teacher_forcing_ratio,
				max_output_len=self.max_output_len,
				freeze_cnn14=self.freeze_cnn14,
				use_cutout=self.use_cutout,
				use_mixup=self.use_mixup,
				use_specaugm=self.use_specaugm,
				beam_alpha=self.beam_alpha,
			)

			self.pred_to_words = Sequential(
				IdxToWords(self.idx_to_word),
				CutSentencesAtTokens(TOKEN_EOS),
				ExcludeWords(TOKEN_SOS, TOKEN_EOS, TOKEN_UNKNOWN),
			)
			self.captions_to_words = self.pred_to_words
	
	def setup(self, stage: Optional[str] = None) -> None:
		if stage in ['fit', None]:
			dataloader = self.trainer.datamodule.train_dataloader()
			audios, audios_lens, _, _ = next(iter(dataloader))
			self.example_input_array = (audios, audios_lens, False)
	
	def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
		audios, audios_lens, captions, captions_lens = batch
		captions[captions == IGNORE_ID] = self.word_to_idx[TOKEN_EOS]

		logits = self.model(audios, audios_lens.cpu(), captions)
		loss = self.criterion(logits, captions[:, 1:].contiguous(), captions_lens - 1)

		with torch.no_grad():
			self.log('train/loss', loss.cpu(), on_epoch=True, on_step=False)

		return loss
	
	def validation_step(self, batch: tuple, batch_idx: int) -> None:
		audios, audios_lens, captions, captions_lens = batch
		captions[captions == IGNORE_ID] = self.word_to_idx[TOKEN_EOS]

		logits = self.model(audios, audios_lens.cpu(), captions)
		loss = self.criterion(logits, captions[:, 1:].contiguous(), captions_lens - 1)

		self.log('val/loss', loss.cpu(), on_epoch=True, on_step=False, prog_bar=True)
	
	def test_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int) -> tuple[list, list]:
		audios, audios_lens, captions, _ = batch
		captions[captions == IGNORE_ID] = self.word_to_idx[TOKEN_EOS]

		preds = self(audios, audios_lens)

		preds = self.pred_to_words(preds)
		captions = self.captions_to_words(captions)
		return preds, captions
	
	def forward(self, audios: Tensor, audios_lens: Tensor, use_beam_search: bool = True) -> list:
		if self.beam_size > 1 and use_beam_search:
			preds = []
			for audio, audio_len in zip(audios, audios_lens):
				audio = audio.unsqueeze(dim=0)
				audio_len = audio_len.unsqueeze(dim=0)
				pred = self.model.beam_search(audio, audio_len)
				preds.append(pred)
		else:
			logits = self.model(audios, audios_lens)
			preds = logits.argmax(dim=-1)
		return preds

	def configure_optimizers(self) -> tuple[list, list]:
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
