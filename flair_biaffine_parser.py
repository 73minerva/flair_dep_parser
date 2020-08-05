
from typing import List, Union, Optional, Callable, Dict

import torch
import torch.nn
from torch.nn.parameter import Parameter

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.embeddings import TokenEmbeddings

from modules.dropout import IndependentDropout, SharedDropout
from modules import BiLSTM, MLP, Biaffine

from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class BiAffineParser(flair.nn.Model):
    def __init__(
            self,
            token_embeddings: TokenEmbeddings,
            relations_dictionary: Dictionary,
            lstm_hidden_size: int = 400,
            n_mlp_arc: int = 500,
            n_mlp_rel: int = 100,
            n_lstm_layers: int = 3,
            dropout: float = 0.3,
            shared_dropout: float = 0.33,
            mlp_dropout: float = .33,
            rnn_layers: int = 2,
            word_dropout: float = 0.05,
            locked_dropout: float = 0.5,
            lstm_dropout: float = 0.2,
            relearn_embeddings: bool = True,
            train_initial_hidden_state: bool = False,
            pickle_module: str = "pickle",

    ):

        super(BiAffineParser, self).__init__()
        self.token_embeddings = token_embeddings

        self.relations_dictionary: Dictionary = relations_dictionary
        print(len(relations_dictionary))
        self.relearn_embeddings = relearn_embeddings

        lstm_input_dim: int = self.token_embeddings.embedding_length

        if self.relations_dictionary:
            self.embedding2nn = torch.nn.Linear(lstm_input_dim, lstm_input_dim)

        self.lstm = BiLSTM(input_size=lstm_input_dim,
                           hidden_size=lstm_hidden_size,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        # self.lstm_shared_dropout = SharedDropout
        # the MLP layers
        self.mlp_arc_h = MLP(n_in=lstm_hidden_size*2,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=lstm_hidden_size*2,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=lstm_hidden_size*2,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=lstm_hidden_size*2,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)

        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=len(relations_dictionary),
                                 bias_x=True,
                                 bias_y=True)
        self.pad_index = "<PAD>"
        self.unk_index = "u"

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.to(flair.device)

    def forward(self, sentences: List[Sentence]):
        self.token_embeddings.embed(sentences)
        batch_size = len(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        seq_len: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.token_embeddings.embedding_length * seq_len,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = seq_len - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.token_embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                batch_size,
                seq_len,
                self.token_embeddings.embedding_length,
            ]
        )

        # --------------------------------------------------------------------
        # BiAffineParser PART
        # --------------------------------------------------------------------
        x = pack_padded_sequence(sentence_tensor, lengths, True, False)

        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        # x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        score_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        score_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        # s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return score_arc, score_rel

    def forward_loss(self, data_points: List[Sentence]) -> torch.tensor:
        # lengths: List[int] = [len(sentence.tokens) for sentence in data_points]

        score_arc, score_rel = self.forward(data_points)
        arc_loss = 0.0
        rel_loss = 0.0

        for sen_id, sen in enumerate(data_points):
            # Root head_id label(arc) returns to 0. We change it in the way that it returns to itself!
            # After this simple change, token indexes doesnt need to start from  1 !

            arc_labels = [token.head_id - 1 if token.head_id != 0 else token.idx - 1 for token in sen.tokens]
            arc_labels = torch.tensor(arc_labels, dtype=torch.int64, device=flair.device)
            arc_loss += self.loss_function(score_arc[sen_id], arc_labels)

            rel_labels = [self.relations_dictionary.get_idx_for_item(token.tags['dependency'].value)
                          for token in sen.tokens]
            rel_labels = torch.tensor(rel_labels, dtype=torch.int64, device=flair.device)
            score_rel = score_rel[sen_id][torch.arange(len(arc_labels)), arc_labels]
            rel_loss += self.loss_function(score_rel, rel_labels)

        return arc_loss+rel_loss
