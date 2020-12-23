import sys
sys.path.insert(0, r"/mnt/transferium/Persoonlijke_Mappen/riwish_hoeseni/slot-filling")

import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from config.global_config import CUDA_PREF, OUTPUT_PREDICTION_DIR, MODEL_OUTPUT_DIR, INPUT_PREDICTION_DIR, \
    PRETRAINED_MODELS_DIR, DATA_DIR, RNN_VOCAB_DOMAIN
from models.rnn.data_loader import get_pretrained_weights, get_vocabulary
from utilities.utils import init_logger, load_tokenizer, get_slot_labels, MODEL_CLASSES_RNN, PRETRAINED_MODEL_PATH

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda:" + str(CUDA_PREF) if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_state_dict(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, model_params, device, vocabulary, weights):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES_RNN[pred_config.model_type](args=pred_config,
                                                      output_size=len(get_slot_labels(pred_config)),
                                                      vocab_size=len(vocabulary['word2index'].keys()),
                                                      weights=weights)
        model.load_state_dict(torch.load(os.path.join(pred_config.model_dir, 'training_args.bin')))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         vocabulary,
                                         pad_token_label_id=-100):
    all_embeddings = []
    all_slot_label_mask = []

    for line in lines:
        embedding = []
        slot_label_mask = []
        line = [e.strip() for e in line.split(" ")]
        for word in line:
            word = str(word).lower()
            # Bad words are equal to vocabulary length
            wordidx = None
            if word not in vocabulary.keys():
                wordidx = len(vocabulary.keys())
            else:
                wordidx = vocabulary[word]
            embedding.append(wordidx)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len([word]) - 1))

        # Zero-pad up to the sequence length.
        embeddings = embedding + ([vocabulary['PAD']] * pred_config.max_seq_len - len(embedding))
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * (pred_config.max_seq_len-len(slot_label_mask)))

        all_embeddings.append(embeddings)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_embeddings = torch.tensor(all_embeddings, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_embeddings, all_slot_label_mask)
    return dataset


def predict(pred_config):
    # load args
    model_params = get_state_dict(pred_config)
    device = get_device(pred_config)
    # load pretrained embeddings and vocabulary
    weight2word, pre_trained_weights = get_pretrained_weights(pred_config)
    vocabulary = None
    try:
        vocabulary = get_vocabulary(pred_config, weight2word)
    except:
        raise Exception("Please note that a vocabulary must be defined prior to performing prediction tasks.")
    # load model
    model = load_model(pred_config, model_params, device, vocabulary, pre_trained_weights)
    logger.info(pred_config)

    slot_label_lst = get_slot_labels(pred_config)

    # Convert input file to TensorDataset
    pad_token_label_id = pred_config.ignore_index
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, vocabulary, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_sentence': batch[0],
                      'slot_labels_ids': None}

            # Change this when implementing different models
            outputs = model(**inputs)
            _, slot_logits = outputs[:2]

            # Slot prediction
            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[1].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[1].detach().cpu().numpy(), axis=0)

    slot_preds = np.argmax(slot_preds, axis=2)
    print(slot_preds)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output_models file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds in zip(lines, slot_preds_list):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            f.write("{}\n".format(line.strip()))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, required=True, type=str, help="Input file for prediction")
    parser.add_argument("--input_file_location", default=INPUT_PREDICTION_DIR, required=False, type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="output_models file for prediction")
    parser.add_argument("--output_file_location", default=OUTPUT_PREDICTION_DIR, required=False, type=str, help="Input file for prediction")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--model_type", default=None, required=True, type=str, help="Model type selected in the list: "
                                                                                    + ", ".join(MODEL_CLASSES_RNN.keys()))
    parser.add_argument("--data_dir", default=DATA_DIR, required=False, type=str, help="The input data dir")
    parser.add_argument("--vocabulary_domain", default=RNN_VOCAB_DOMAIN, required=False, type=str, help="The input data dir")
    parser.add_argument("--model_embedding", default=None, required=False, type=str,
                        help="Use one of the following pre-trained embeddings or leave blank to use default nn.Embeddings" + ", ".join(
                            [e for e in PRETRAINED_MODEL_PATH if 'word2vec' in e.lower()]))
    parser.add_argument("--max_seq_len", default=300, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    pred_config.input_file = os.path.join(INPUT_PREDICTION_DIR, pred_config.input_file)
    pred_config.output_file = os.path.join(OUTPUT_PREDICTION_DIR, pred_config.output_file)
    pred_config.model_dir = os.path.join(MODEL_OUTPUT_DIR, pred_config.model_dir)
    pred_config.model_embedding = PRETRAINED_MODEL_PATH[pred_config.model_embedding] if pred_config.model_embedding else None
    pred_config.model_embedding_path = os.path.join(PRETRAINED_MODELS_DIR, pred_config.model_embedding) if pred_config.model_embedding else None
    pred_config.task = pred_config.vocabulary_domain

    predict(pred_config)
