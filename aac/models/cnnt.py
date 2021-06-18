
import os.path as osp
import torch

from torch import Tensor, nn
from torch.nn import Module
from typing import Optional

from aac.models.pann.cnn14_decision import Cnn14_DecisionLevelAtt 
from aac.models.pann.urls import PANN_PRETRAINED_URLS
from aac.models.lat_decoder_v2 import BeamDecoderV2


class CnnTell(Module):
	"""
		CNN14 (Cnn14_DecisionLevelAtt embeddings) as encoder + "Teller" (ListenAttendTell decoder) as decoder
	"""
	def __init__(
		self,
		dpath_pretrained: Optional[str],
		word_to_idx: dict[str, int],
		n_classes: int,
		embedding_dim: int = 128,
		decoder_hidden_size_1: int = 128,
		decoder_hidden_size_2: int = 64,
		query_size: int = 64,
		value_size: int = 64,
		key_size: int = 64,
		is_attended: bool = True,
		beam_size: int = 10,
		teacher_forcing_ratio: float = 0.98,
		max_output_len: int = 30,
		freeze_cnn14: bool = True,
		use_cutout: bool = False,
		use_mixup: bool = False,
		use_specaugm: bool = False,
		beam_alpha: float = 1.2,
	) -> None:
		super().__init__()

		# TODO : change the hard-coded values
		self.encoder = Cnn14_DecisionLevelAtt(
			sample_rate=32000, 
			window_size=1024, 
			hop_size=320, 
			mel_bins=64, 
			fmin=50, 
			fmax=14000, 
			classes_num=527,
			use_cutout=use_cutout,
			use_mixup=use_mixup,
			use_spec_augment=use_specaugm,
		)

		embed_size = 2048  # defined in Cnn14_DecisionLevelAtt
		self.fc_key = nn.Linear(embed_size, key_size)
		self.fc_value = nn.Linear(embed_size, value_size)

		self.decoder = BeamDecoderV2(
			vocab_size=n_classes, 
			word2index=word_to_idx,
			embedding_dim=embedding_dim,
			decoder_hidden_size_1=decoder_hidden_size_1,
			decoder_hidden_size_2=decoder_hidden_size_2,
			query_size=query_size,
			value_size=value_size,
			key_size=key_size,
			is_attended=is_attended, 
			beam_size=beam_size, 
			teacher_forcing_ratio=teacher_forcing_ratio, 
			max_output_len=max_output_len,
			beam_alpha=beam_alpha,
		)

		if dpath_pretrained is not None:
			if not osp.isdir(dpath_pretrained):
				raise RuntimeError(f'Pre-trained path "{dpath_pretrained}" is not a directory.')
			fpath_cnn14 = osp.join(dpath_pretrained, PANN_PRETRAINED_URLS['Cnn14_DecisionLevelAtt']['fname'])
			if not osp.isfile(fpath_cnn14):
				raise RuntimeError(f'Cannot use pre-trained weights for Cnn14_DecisionLevelAtt (fpath "{fpath_cnn14}" is not a file).')
			data = torch.load(fpath_cnn14, map_location=torch.device('cpu'))
			self.encoder.load_state_dict(data['model'], strict=False)
		
		if freeze_cnn14:
			for param in self.encoder.parameters():
				param.requires_grad = False

	def forward(
		self, 
		audios: Tensor, 
		audios_lens: Tensor, 
		captions: Optional[Tensor] = None,
	) -> Tensor:
		"""
			:param audios: (bsize, audio_len)
			:param audios_lens: (bsize,)
			:param captions: (bsize, sentence_len)
		"""
		if len(audios.shape) != 2:
			raise RuntimeError(f'Model "{self.__class__.__name__}" expects raw audio batch tensor of shape (bsize, audio_len), but found shape {audios.shape}.')

		audio_len = audios.shape[1]
		device = audios.device
		
		encoder_outputs = self.encoder(audios)

		framewise_embed = encoder_outputs['framewise_embed']
		# (bsize, embed_len, n_frames) -> (n_frames, bsize, embed_len)
		framewise_embed = framewise_embed.permute(2, 0, 1)
		n_frames = framewise_embed.shape[0]

		key = self.fc_key(framewise_embed)
		value = self.fc_value(framewise_embed)

		reduction_factor = audio_len // n_frames
		out_encoder_lengths = (audios_lens // reduction_factor).to(device=device)
		out_encoder_lengths = out_encoder_lengths.unsqueeze(1)
		out_encoder_T = audio_len // reduction_factor
		indices = torch.arange(0, out_encoder_T, device=device).unsqueeze(0)
		mask_encoder_output = indices < out_encoder_lengths
		mask_encoder_output = mask_encoder_output.unsqueeze(1)  # B, 1, T_after_pBLSTM_reduction

		logits = self.decoder(
			key, 
			value, 
			mask_encoder_output, 
			text=captions, 
			isTrain=captions is not None,
			return_attention_masks=False, 
			use_gumbel_noise=False,
		)
		
		# (bsize, caption_len, vocab_len)
		return logits
	
	def beam_search(
		self, 
		audios: Tensor, 
		audios_lens: Tensor, 
	) -> list:
		"""
			:param audios: (bsize, audio_len)
			:param audios_lens: (bsize,)
			:param captions: (bsize, sentence_len)
		"""
		if len(audios.shape) != 2:
			raise RuntimeError(f'Model "{self.__class__.__name__}" expects raw audio batch tensor of shape (bsize, audio_len), but found shape {audios.shape}.')

		audio_len = audios.shape[1]
		device = audios.device
		
		encoder_outputs = self.encoder(audios)

		framewise_embed = encoder_outputs['framewise_embed']
		# (bsize, embed_len, n_frames) -> (n_frames, bsize, embed_len)
		framewise_embed = framewise_embed.permute(2, 0, 1)
		n_frames = framewise_embed.shape[0]

		key = self.fc_key(framewise_embed)
		value = self.fc_value(framewise_embed)

		reduction_factor = audio_len // n_frames
		out_encoder_lengths = (audios_lens // reduction_factor).to(device=device)
		out_encoder_lengths = out_encoder_lengths.unsqueeze(1)
		out_encoder_T = audio_len // reduction_factor
		indices = torch.arange(0, out_encoder_T, device=device).unsqueeze(0)
		mask_encoder_output = indices < out_encoder_lengths
		mask_encoder_output = mask_encoder_output.unsqueeze(1)  # B, 1, T_after_pBLSTM_reduction

		logits = self.decoder.beam_search(
			key, 
			value, 
			mask_encoder_output, 
			text=None, 
			isTrain=False,
			return_attention_masks=False, 
			use_gumbel_noise=False,
		)
		
		# (bsize, caption_len, vocab_len)
		return logits
