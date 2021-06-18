
import torch
import torch.nn.functional as F_torch

from torch import Tensor
from typing import Optional

from aac.models.listen_attend_tell import Decoder


class BeamDecoderV2(Decoder):
	""" 
		ListenAttendTell decoder with beam_search method. 
		IMPORTANT : The forward does a standard greedy search and returns logits for training. 
	"""
	def __init__(
		self,
		vocab_size: int, 
		word2index: dict[str, int], 
		embedding_dim: int = 128, 
		decoder_hidden_size_1: int = 128, 
		decoder_hidden_size_2: int = 64,
		query_size: int = 64, 
		value_size: int = 64, 
		key_size: int = 64, 
		is_attended: bool = True, 
		beam_size: int = 10, 
		teacher_forcing_ratio: float = 0.9, 
		max_output_len: int = 30,
		beam_alpha: float = 1.2,
	) -> None:
		super().__init__(
			vocab_size=vocab_size,
			embedding_dim=embedding_dim,
			decoder_hidden_size_1=decoder_hidden_size_1, 
			decoder_hidden_size_2=decoder_hidden_size_2,
			query_size=query_size, 
			value_size=value_size, 
			key_size=key_size, 
			emb_fpath=None, 
			freeze_embeddings=False,
			isAttended=is_attended, 
			teacher_forcing_ratio=teacher_forcing_ratio, 
			word2index=word2index,  
			max_output_len=max_output_len,
		)
		# Params
		self.vocab_size = vocab_size
		self.word2index = word2index
		self.embedding_dim = embedding_dim
		self.decoder_hidden_size_1 = decoder_hidden_size_1
		self.decoder_hidden_size_2 = decoder_hidden_size_2
		self.query_size = query_size
		self.value_size = value_size
		self.key_size = key_size
		self.is_attended = is_attended
		self.beam_size = beam_size
		self.teacher_forcing_ratio = teacher_forcing_ratio
		self.max_output_len = max_output_len
		self.beam_alpha = beam_alpha
	
	def beam_search(
		self, 
		key: Tensor, 
		values: Tensor, 
		mask: Tensor, 
		return_attention_masks: bool = False,
		**kwargs,
	) -> list[int]:
		"""
			Note : this version of beam search accept only batch size B = 1

			:param key :(T, B, key_size) Output of the Encoder Key projection layer
			:param values: (T, B, value_size) Output of the Encoder Value projection layer
			:param mask: (B, 1, T) be careful, B the batch size is first dim! Useful for attention
			:return hypothesis: The best hypothesis
		"""
		assert key.shape[1] == values.shape[1] == mask.shape[0] == 1, f'Bsize != 1 is not supported by {self.__class__.__name__}.'

		device = key.device
		k = self.beam_size
		# at word-level, 30 words for audio captioning should be enough
		max_output_len = self.max_output_len  

		key = key.expand(-1, k, -1)
		values = values.expand(-1, k, -1)
		mask = mask.expand(k, -1, -1)

		hidden_states: list[Optional[tuple]] = [None, None]

		# We'll treat the problem as having a batch size of k
		k_prev_words = torch.zeros(k, dtype=torch.long, device=device)

		# initiating with '<sos>'
		k_prev_words[:] = self.word2index['<sos>']
		# prediction[:, -1] = 1. # for unit tests

		# Tensor to store top k sequences' scores; now they're just 0
		top_k_scores = torch.zeros(k, 1, device=device)  # (k, 1)

		# Tensor to store top k sequences; now they're just <sos>
		seqs = k_prev_words.unsqueeze(1)  # (k, 1)

		# Lists to store completed sequences and scores
		complete_seqs = []
		complete_seqs_scores = []

		context = torch.zeros(k, self.query_size, device=device)  # initialize context to 0 for the first prediction

		hypotheses = []
		att_masks = []

		for i in range(max_output_len):
			# char_embed = self.embedding(prev_token.argmax(dim=-1))
			char_embed = self.embedding(k_prev_words)

			inp = torch.cat([char_embed, context], dim=1)  # context is size (B, h)
			hidden_states[0] = self.lstm1(inp, hidden_states[0])  # outputs a tuple (next_hidden_state, next_cell_state)

			inp_2 = hidden_states[0][0]
			hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

			output = hidden_states[1][0]  # B, hidden//2, ex: B, 256

			if self.isAttended:
				### Compute attention from the output of the second LSTM Cell ###
				query = self.query_network(output)  # B, h
				# key and values are fixed and are the output of the encoder, size: T,B,h

				# print(f'DEBUG latV1: query={query.shape}; key={key.shape}; values={values.shape}; mask={mask.shape}')
				context, attention_mask = self.attention(query, key, values, mask)  # context: B,h ; attention_mask: B, T_speech_after_pBLSTM_reduction

				# if k < 2:
				# 	context = torch.unsqueeze(context, dim=0)
				if return_attention_masks:
					att_masks.append(attention_mask.unsqueeze(1))

				prediction = self.character_prob(
					torch.cat([output, context], dim=1)
				)  # [B, hidden] concat with [B, key_hidden] ---> linear layer -->  B, 
														   
			else:
				# use values at time step i instead of context
				prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))  # B, vocab

			cur_prob = F_torch.log_softmax(prediction, dim=1)

			# Add
			scores = top_k_scores.expand_as(cur_prob) + cur_prob  # (s, vocab_size)

			# For the first step, all k points will have the same scores (since same k previous words, h, c)
			if i == 0:
				top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
			else:
				# Unroll and find top scores, and their unrolled indices
				top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

			# Convert unrolled indices to actual indices of scores
			prev_word_inds = top_k_words // self.vocab_size  # (s)
			next_word_inds = top_k_words % self.vocab_size  # (s)

			# Add new words to sequences
			seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

			# Which sequences are incomplete (didn't reach <eos>)?
			incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word2index['<eos>']]
			complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

			# if we reached max_output_len, add <eos> to incomplete sentences
			if i == max_output_len - 1:
				tmp = []
				for _ in range(len(seqs)):
					tmp.append(self.word2index['<eos>'])
				next_word_inds = torch.as_tensor(tmp, device=device, dtype=torch.long)
				seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)],
								dim=1)  # (s, step+1)  # (s, step+1)

				# Which sequences are incomplete (didn't reach <eos>)?
				incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
								next_word != self.word2index['<eos>']]
				complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

			# Set aside complete sequences
			if len(complete_inds) > 0:
				complete_seqs.extend(seqs[complete_inds].tolist())
				nb_of_words_complete_seqs = [len(caption) for caption in seqs[complete_inds].tolist()]
				raw_scores = top_k_scores[complete_inds]
				complete_seqs_scores.extend([1. / (nb_of_words_complete_seqs[ind_seq] ** self.beam_alpha) * sc for ind_seq, sc in
											 enumerate(raw_scores.tolist())])

			k -= len(complete_inds)  # reduce beam length accordingly

			# Proceed with incomplete sequences
			if k == 0:
				break
			seqs = seqs[incomplete_inds]
			hidden_states[0] = (
				hidden_states[0][0][prev_word_inds[incomplete_inds]],
				hidden_states[0][1][prev_word_inds[incomplete_inds]]
			)
			hidden_states[1] = (
				hidden_states[1][0][prev_word_inds[incomplete_inds]],
				hidden_states[1][1][prev_word_inds[incomplete_inds]]
			)
			context = context[prev_word_inds[incomplete_inds]]

			key = key[:, :k, :]
			values = values[:, :k, :]
			mask = mask[:k, :, :]

			top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
			k_prev_words = next_word_inds[incomplete_inds]

		i = complete_seqs_scores.index(max(complete_seqs_scores))
		seq = complete_seqs[i]
		hypotheses.append([w for w in seq if w not in {self.word2index['<sos>'], self.word2index['<eos>']}])

		return hypotheses[0]
