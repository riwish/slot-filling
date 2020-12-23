import os
import copy
import json
import logging

import torch
import gensim
from torch.utils.data import TensorDataset

from config.global_config import TRAIN_FOLDER_NAME, VAL_FOLDER_NAME, TEST_FOLDER_NAME
from utilities.utils import get_slot_labels, load_pickle, save_pickle

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, slot_labels=None):
        self.guid = guid
        self.words = words
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, embedding, slot_labels_ids):
        self.embedding = embedding
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """Processor for ATIS look-a-like data sets """

    def __init__(self, args):
        self.args = args
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, slots, set_type):
        """Creates examples for the training and val sets."""
        examples = []
        for i, (text, slot) in enumerate(zip(texts, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()
            # 2. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, val, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


processors = {
    "atis": Processor,
    "snips": Processor,
    "dummy": Processor,
    "dummy_masked": Processor,
    "dummy_aug": Processor,
    "dummy_masked_aug": Processor,
}


def get_vocabulary(args, weight2word=None):
    # Load vocabulary from cache file
    cached_vocab_file = os.path.join(
        args.data_dir,
        'cached_vocab_{}_{}_{}'.format(
            args.task,
            args.model_type,
            args.model_embedding
        )
    )

    if os.path.exists(cached_vocab_file):
        logger.info("Loading vocab from cached file %s", cached_vocab_file)
        vocab = load_pickle(cached_vocab_file)
        return vocab
    else:
        processor = processors[args.task](args)
        examples = processor.get_examples(TRAIN_FOLDER_NAME)

        word2index = {}
        # Fill vocabulary with pretrained words first
        if weight2word:
            for idx, word in enumerate(weight2word):
                word2index[word] = idx

        special_tokens = ['PAD', 'UNK', 'MASK']

        # Fill vocabulary with special words first
        for idx, word in enumerate(special_tokens):
            word2index[word] = len(word2index)

        # Fill vocabulary with domain data words
        for sample in examples:
            for word in sample.words:
                word = str(word).lower()
                if word not in word2index.keys():
                    word2index[word] = len(word2index)
        idx2word = dict(zip(word2index.values(), word2index.keys()))
        vocab = {'word2index': word2index, 'index2word': idx2word}
        save_pickle(data=vocab, filename=cached_vocab_file)
        return vocab


def get_pretrained_weights(args):
    if 'word2vec' in str(args.model_embedding).lower():
        pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(args.model_embedding_path, 'model.bin'), binary=True)
        return list(pretrained_model.index2word), torch.FloatTensor(pretrained_model.vectors)
    elif 'glove' in str(args.model_embedding).lower():
        # TODO: experiment with GloVe vector embeddings
        return None, None
    else:
        logger.info("No pretrained embeddings will be used")
        return None, None


def convert_examples_to_features(examples, max_seq_len, vocabulary,
                                 pad_token_label_id=-100):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        embedding = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word = str(word).lower()
            # Bad words are equal to vocabulary length
            wordidx = None
            if word not in vocabulary.keys():
                wordidx = len(vocabulary.keys())-1
            else:
                wordidx = vocabulary[word]
            embedding.append(wordidx)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len([word]) - 1))

        embedding = embedding + ([vocabulary['PAD']] * (max_seq_len - len(embedding)))
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * (max_seq_len-len(slot_labels_ids)))

        assert len(embedding) == max_seq_len, "Error with embedding length {} vs {}".format(len(embedding), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("Embedding: %s" % embedding)
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(embedding=embedding,
                          slot_labels_ids=slot_labels_ids
                          ))
    return features


def load_and_cache_examples(args, vocabulary, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_{}'.format(
            mode,
            args.task,
            args.model_type,
            args.max_seq_len,
            args.model_embedding
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == TRAIN_FOLDER_NAME:
            examples = processor.get_examples(TRAIN_FOLDER_NAME)
        elif mode == VAL_FOLDER_NAME:
            examples = processor.get_examples(VAL_FOLDER_NAME)
        elif mode == TEST_FOLDER_NAME:
            examples = processor.get_examples(TEST_FOLDER_NAME)
        else:
            raise Exception("For mode, Only train, val, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, vocabulary,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_embeddings = torch.tensor([f.embedding for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_embeddings, all_slot_labels_ids)
    return dataset