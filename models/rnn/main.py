import sys
sys.path.insert(0, r"/mnt/transferium/Persoonlijke_Mappen/riwish_hoeseni/slot-filling")

import argparse
from models.rnn.trainer import Trainer
from utilities.utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES_RNN, PRETRAINED_MODEL_PATH, create_batch
from models.rnn.data_loader import get_vocabulary, get_pretrained_weights, load_and_cache_examples
from config.global_config import *


def main(args):
    init_logger()
    set_seed(args)

    # Get pretrained embeddings and vocabulary
    weight2word, pre_trained_weights = get_pretrained_weights(args)
    vocabulary = get_vocabulary(args, weight2word)
    # Load datasets
    train_set = load_and_cache_examples(args, vocabulary=vocabulary['word2index'], mode=TRAIN_FOLDER_NAME)
    val_set = load_and_cache_examples(args, vocabulary=vocabulary['word2index'], mode=VAL_FOLDER_NAME)
    test_set = load_and_cache_examples(args, vocabulary=vocabulary['word2index'], mode=TEST_FOLDER_NAME)
    trainer = Trainer(args, train_set, val_set, test_set, vocabulary, pre_trained_weights)

    if args.do_train:
        trainer.train()
    if args.do_eval or args.do_baseline_eval:
        if args.do_eval:
            # Evaluate with a neural model only
            trainer.load_model()
            trainer.evaluate_neural(TEST_FOLDER_NAME)
        elif args.do_baseline_eval:
            # Evaluate with a neural model only
            trainer.evaluate_baseline(TEST_FOLDER_NAME, args)
    elif not args.do_train and not args.do_eval and args.do_baseline_eval:
        raise Exception("No specific instruction has been given what to do with the model. Please specify whether to "
                        "train and/or evaluate the model. See --help for more information.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main arguments
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default=DATA_DIR, required=False, type=str, help="The input data dir")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Provide foldername to save model")
    parser.add_argument("--model_type", default=None, required=True, type=str, help="Model type selected in the list: "
                                                                                    + ", ".join(MODEL_CLASSES_RNN.keys()))
    parser.add_argument("--model_embedding", default=None, required=False, type=str,
                        help="Use one of the following pre-trained embeddings or leave blank to use default nn.Embeddings" + ", ".join(
                            [e for e in PRETRAINED_MODEL_PATH if 'word2vec' in e.lower()]))


    # Train and/or Evaluate
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run an evaluation on the test set.")
    parser.add_argument("--do_baseline_eval", action="store_true", help="Whether to run a basline evaluation on the test set.")

    # Logging arguments
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    # Optional arguments
    parser.add_argument("--slot_label_file", default=SLOT_ANNOTATIONS_FILENAME, type=str, help="Slot Label annotation file")
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")

    # Batch size arguments
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")

    # Model Hyperparameters arguments
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--hidden_size", default=256, type=float, help="The size of the hidden state of the RNN")
    parser.add_argument("--num_layers", default=2, type=float, help="The number of layers within the RNN")

    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')
    parser.add_argument("--max_seq_len", default=150, type=int,
                        help="The maximum total input sequence length after tokenization.")

    # GPU arguments
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--use_cuda_device", default=True, type=bool, help="Use GPU device.")

    args = parser.parse_args()
    args.model_dir = os.path.join(MODEL_OUTPUT_DIR, args.model_dir)
    args.model_embedding = PRETRAINED_MODEL_PATH[args.model_embedding] if args.model_embedding else None
    args.model_embedding_path = os.path.join(PRETRAINED_MODELS_DIR, args.model_embedding) if args.model_embedding else None

    args.do_eval = False if args.do_baseline_eval else args.do_eval
    main(args)
