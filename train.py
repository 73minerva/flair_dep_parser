from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.datasets import UniversalDependenciesCorpus
from flair.data import Dictionary, Corpus
from flair_parser.flair_biaffine_parser.flair_biaffine_parser import BiAffineParser
from flair.models import sequence_tagger_model

from trainer import ModelTrainer
import flair
import torch

flair.device = torch.device('cpu')


def make_relations_tag_dictionary(corpus: Corpus, tag_type='dependency', special_tags=[]) -> Dictionary:

    tag_dictionary: Dictionary = Dictionary(add_unk=False)
    # for tag in special_tags:
    #     tag_dictionary.add_item(tag)
    for sentence in corpus.get_all_sentences():
        for token in sentence.tokens:
            tag_dictionary.add_item(token.get_tag(tag_type).value)
    return tag_dictionary


if __name__ == '__main__':
    embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),

    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    corpus: UniversalDependenciesCorpus = UniversalDependenciesCorpus(data_folder="./data/")

    relations_dictionary = make_relations_tag_dictionary(corpus)
    print(relations_dictionary)

    dep_parse: BiAffineParser = BiAffineParser(embeddings, relations_dictionary)

    trainer: ModelTrainer = ModelTrainer(dep_parse, corpus)

    # 7. start training
    trainer.train('./biaffine_dep_parser/',
                  learning_rate=0.1,
                  mini_batch_size=1)