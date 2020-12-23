import os
import random
import pickle
import itertools
import logging

import torch
import spacy
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer

from config.global_config import HOME_DIR
from models.rnn.model import BiRNN
from models.transformer.model import TransformerBert, TransformersRobbert


# Define new RNN configurations here
MODEL_CLASSES_RNN = {
    'BLSTM': BiRNN
}

# Define new transformer configurations here
MODEL_CLASSES_TRANSFORMERS = {
    'transformer_bert': (BertConfig, TransformerBert, BertTokenizer),
    'transformer_bert-multi': (BertConfig, TransformerBert, BertTokenizer),
    'transformer_bertje': (BertConfig, TransformerBert, BertTokenizer),
    'transformer_roberta': (RobertaConfig, TransformersRobbert, RobertaTokenizer),
    'transformer_robbert': (RobertaConfig, TransformersRobbert, RobertaTokenizer)
}

# Link transformer model type to pretrained model folder name
PRETRAINED_MODEL_PATH = {
    'bert': 'bert-base-uncased',
    'bert-multi': 'bert-base-multilingual-uncased',
    'bertje': 'bert-base-dutch-cased',
    'roberta': 'roberta-base',
    'robbert': 'robbert-base',
    'word2vec_dutch': 'word2vec_dutch_300'
}


def load_knowledge():
    return load_pickle(os.path.join(HOME_DIR, 'models', 'baseline', 'resources', 'knowledge.pickle'))


def load_pickle(filename):
    with open(os.path.join(filename), 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, filename):
    with open(os.path.join(filename), 'wb') as f:
        pickle.dump(data, f)


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES_TRANSFORMERS[args.model_type][2].from_pretrained(args.pretrained_model_path)


def create_batch(iterable, n):
    iterable=iter(iterable)
    while True:
        chunk=[]
        for i in range(n):
            try:
                chunk.append(next(iterable))
            except StopIteration:
                yield chunk
                return
        yield chunk


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    slot_result_per_label = get_slot_metrics_per_label(slot_preds, slot_labels)
    semantic_result = get_sentence_frame_acc(slot_preds, slot_labels)

    results.update(slot_result)
    results.update({'label_metrics': slot_result_per_label})
    results.update(semantic_result)
    return results


def get_slot_metrics_per_label(preds, labels):
    assert len(preds) == len(labels)
    unique_target_labels = list(set(list(itertools.chain(*labels))))

    metric_dict = {l: {} for l in unique_target_labels}
    for target, metrics in metric_dict.items():
        tp = 0
        fp = 0
        fn = 0
        for y_true, y_pred in zip(labels, preds):
            # Convert to numpy for faster computation
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            # Get indices of target label
            y_true_indices_target = list(np.where(y_true == target)[0])
            y_pred_indices_target = list(np.where(y_pred == target)[0])

            if y_pred_indices_target == y_true_indices_target:
                tp += len(y_pred_indices_target)
            elif not y_pred_indices_target:
                # Not being able to predict the target at the correct index counts for FN
                fn += len(y_true_indices_target)
            else:
                # Get TP first
                tp += len(list(set(y_true_indices_target) & set(y_pred_indices_target)))
                # Get FP (difference in positions from y_pred to y_true)
                fp += len(list(set(y_pred_indices_target) - set(y_true_indices_target)))

        # Calculate final metrics
        precision = tp / (tp + fp) if tp !=0 else 0
        recall = tp / (tp + fn) if tp !=0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision !=0 else 0
        metric_dict[target]['label_precision'] = precision
        metric_dict[target]['label_recall'] = recall
        metric_dict[target]['label_f1'] = f1
    return metric_dict


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(slot_preds, slot_labels):
    """For the cases that all the slots are correct (in one sentence)"""
    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    # Calculate percentage from Boolean list
    semantic_acc = float(np.sum(slot_result)/slot_result.shape[0])
    return {
        "semantic_frame_acc": semantic_acc
    }
