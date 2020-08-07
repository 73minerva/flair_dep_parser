from pathlib import Path
from typing import List, Union, Optional, Callable, Dict, Tuple

import torch
import torch.nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.embeddings import TokenEmbeddings
from flair.training_utils import Metric, Result, store_embeddings

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
            mlp_dropout: float = .33,
            lstm_dropout: float = 0.2,
            relearn_embeddings: bool = True,
            beta: float = 1.0,
            pickle_module: str = "pickle",
    ):

        super(BiAffineParser, self).__init__()
        self.token_embeddings = token_embeddings
        self.beta = beta
        self.relations_dictionary: Dictionary = relations_dictionary
        self.relearn_embeddings = relearn_embeddings
        self.lstm_hidden_size = lstm_hidden_size
        self.n_mlp_arc = n_mlp_arc
        self.n_mlp_rel = n_mlp_rel
        self.n_lstm_layers = n_lstm_layers
        self.lstm_dropout = lstm_dropout
        self.mlp_dropout = mlp_dropout
        lstm_input_dim: int = self.token_embeddings.embedding_length

        if self.relations_dictionary:
            self.embedding2nn = torch.nn.Linear(lstm_input_dim, lstm_input_dim)

        self.lstm = BiLSTM(input_size=self.lstm_input_dim,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.n_lstm_layers,
                           dropout=self.lstm_dropout)

        self.mlp_arc_h = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.n_mlp_arc,
                             dropout=self.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.n_mlp_arc,
                             dropout=self.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.n_mlp_rel,
                             dropout=self.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.lstm_hidden_size*2,
                             n_hidden=self.n_mlp_rel,
                             dropout=self.mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=self.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)

        self.rel_attn = Biaffine(n_in=self.n_mlp_rel,
                                 n_out=len(relations_dictionary),
                                 bias_x=True,
                                 bias_y=True)

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

        score_arc, score_rel = self.forward(data_points)
        loss_arc, loss_rel = self._calculate_loss(score_arc, score_rel, data_points)
        main_loss = loss_arc + loss_rel

        return main_loss

    def _calculate_loss(self, score_arc: torch.tensor,
                        score_relation: torch.tensor,
                        data_points: List[Sentence]) -> Tuple[float, float]:

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
            score_relation = score_relation[sen_id][torch.arange(len(arc_labels)), arc_labels]
            rel_loss += self.loss_function(score_relation, rel_labels)

        return arc_loss, rel_loss



    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            "token_embeddings": self.token_embeddings,
            "lstm_hidden_size": self.lstm_hidden_size,
            "relations_dictionary": self.relations_dictionary,
            "relearn_embeddings": self.relearn_embeddings,
            "n_mlp_arc": self.n_mlp_arc,
            "n_mlp_rel": self.n_mlp_rel,
            "n_lstm_layers": self.n_lstm_layers,
            "lstm_dropout": self.lstm_dropout,
            "mlp_dropout": self.mlp_dropout,
            "beta": self.beta,
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        model = BiAffineParser(
            token_embeddings=state["token_embeddings"],
            relations_dictionary=state["relations_dictionary"],
            lstm_hidden_size=state["lstm_hidden_size"],
            n_mlp_arc=state["n_mlp_arc"],
            n_mlp_rel=state["n_mlp_rel"],
            n_lstm_layers=state["n_lstm_layers"],
            mlp_dropout=state["mlp_dropout"],
            lstm_dropout=state["lstm_dropout"],
            relearn_embeddings=state["relearn_embeddings"],
            beta=state["beta"],
        )
        model.load_state_dict(state["state_dict"])
        return model

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embedding_storage_mode: str = "none",
    ) -> (Result, float):

        if type(out_path) == str:
            out_path = Path(out_path)
        metric = Metric("Evaluation", beta=self.beta)
        parsing_metric = ParsingMetric()

        lines: List[str] = []

        eval_loss_arc = 0
        eval_loss_rel = 0

        for batch_idx, batch in enumerate(data_loader):

            with torch.no_grad():
                score_arc, score_rel = self.forward(batch)
                loss_arc, loss_rel = self._calculate_loss(score_arc, score_rel, batch)
                arc_prediction, relation_prediction = self._obtain_labels_(score_arc, score_rel)

            parsing_metric(arc_prediction, relation_prediction, batch)

            eval_loss_arc += loss_arc
            eval_loss_rel += loss_rel

            for (sentence, arcs, sent_tags) in zip(batch, arc_prediction, relation_prediction):
                for (token, arc, tag) in zip(sentence.tokens, arcs, sent_tags):
                    token: Token = token
                    token.add_tag_label("predicted", Label(tag))
                    token.add_tag_label("predicted_head_id", Label(str(arc)))

                    # append both to file for evaluation
                    eval_line = "{} {} {} {} {}\n".format(
                        token.text,
                        token.tags['dependency'].value,
                        str(token.head_id),
                        tag,
                        str(arc),
                    )
                    lines.append(eval_line)
                lines.append("\n")

            for sentence in batch:

                # make list of gold tags
                gold_tags = [token.tags['dependency'].value for token in sentence.tokens]

                # make list of predicted tags
                predicted_tags = [tag.tag for tag in sentence.get_spans("predicted")]

                # check for true positives, false positives and false negatives
                for tag_indx, predicted_tag in enumerate(predicted_tags):
                    if predicted_tag == gold_tags[tag_indx]:
                        metric.add_tp(tag)
                    else:
                        metric.add_fp(tag)

                for tag_indx, label_tag in enumerate(gold_tags):
                    if label_tag != predicted_tags[tag_indx]:
                        metric.add_fn(tag)
                    else:
                        metric.add_tn(tag)
            store_embeddings(batch, embedding_storage_mode)

        eval_loss_arc /= len(data_loader)
        eval_loss_rel /= len(data_loader)

        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.write("".join(lines))

        detailed_result = (
            f"\nUAS : {parsing_metric.get_uas():.4f} - LAS : {parsing_metric.get_las():.4f}"
            f"\neval loss rel : {eval_loss_rel:.4f} - eval loss arc : {eval_loss_arc:.4f}"
            f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
            f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
        )
        for class_name in metric.get_classes():
            detailed_result += (
                f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                f"{metric.f_score(class_name):.4f}"
            )

        result = Result(
            main_score=metric.micro_avg_f_score(),
            log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
            log_header="PRECISION\tRECALL\tF1",
            detailed_results=detailed_result,
        )

        return result, eval_loss_arc+eval_loss_rel

    def _obtain_labels_(self, score_arc: torch.tensor, score_rel: torch.tensor) -> Tuple[List[List[int]],
                                                                                         List[List[str]]]:
        arc_prediction: torch.tensor = score_arc.argmax(-1)
        relation_prediction: torch.tensor = score_rel.argmax(-1)
        relation_prediction = relation_prediction.gather(-1, arc_prediction.unsqueeze(-1)).squeeze(-1)

        arc_prediction = [[arc+1 if token_index != arc else 0 for token_index, arc in enumerate(batch)]
                          for batch in arc_prediction]
        relation_prediction = [[self.relations_dictionary.get_item_for_index(rel_tag_idx)
                                for rel_tag_idx in batch] for batch in relation_prediction]

        return arc_prediction, relation_prediction


class ParsingMetric:

    def __init__(self, epsilon=1e-6):
        self.eps = epsilon
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __call__(self, arc_prediction: List[List[int]],
                 relation_prediction: List[List[str]],
                 sentences: List[Sentence]):

        for batch_indx, batch in enumerate(sentences):
            self.total += len(batch.tokens)
            for token_indx, token in enumerate(batch.tokens):
                if arc_prediction[batch_indx][token_indx] == token.head_id:
                    self.correct_arcs += 1
                if relation_prediction[batch_indx][token_indx] == token.tags['dependency'].value:
                    self.correct_rels += 1

    def get_las(self) -> float:
        return self.correct_rels / (self.total + self.eps)

    def get_uas(self) -> float:
        return self.correct_arcs / (self.total + self.eps)

