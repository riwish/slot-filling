import sys
sys.path.insert(0, r"/mnt/transferium/Persoonlijke_Mappen/riwish_hoeseni/slot-filling")

import os
import argparse
from models.transformer.trainer import Trainer
from config.global_config import DATA_DIR, CUDA_PREF, PRETRAINED_MODELS_DIR, SLOT_ANNOTATIONS_FILENAME, MODEL_OUTPUT_DIR, TEST_FOLDER_NAME, TRAIN_FOLDER_NAME, VAL_FOLDER_NAME
from utilities.utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES_TRANSFORMERS, PRETRAINED_MODEL_PATH
from models.transformer.data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)

    if args.do_train:
        tokenizer = load_tokenizer(args)
        train_set = load_and_cache_examples(args, tokenizer, mode=TRAIN_FOLDER_NAME)
        val_set = load_and_cache_examples(args, tokenizer, mode=VAL_FOLDER_NAME)
        test_set = load_and_cache_examples(args, tokenizer, mode=TEST_FOLDER_NAME)
        trainer = Trainer(args, train_set, val_set, test_set)
        trainer.train()

        if args.do_eval:
            # Evaluate with a neural model only
            trainer.load_model()
            trainer.evaluate_neural(TEST_FOLDER_NAME)
        elif args.do_baseline_eval:
            # Evaluate with a neural model only
            trainer.evaluate_baseline(TEST_FOLDER_NAME)

    elif args.do_eval or args.do_baseline_eval:
        if args.do_eval:
            tokenizer = load_tokenizer(args)
            test_set = load_and_cache_examples(args, tokenizer, mode=TEST_FOLDER_NAME)
            trainer = Trainer(args, None, None, test_set)
            trainer.load_model()
            # Evaluate with a neural model only
            trainer.evaluate_neural(TEST_FOLDER_NAME)
        else:
            trainer = Trainer(args, None, None, None)
            # Evaluate with a static model only
            trainer.evaluate_baseline(TEST_FOLDER_NAME, args)
    else:
        raise Exception("No specific instruction has been given what to do with the model. Please specify whether to "
                        "train and/or evaluate the model. See --help for more information.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Main arguments
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default=DATA_DIR, required=False, type=str, help="The input data dir")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--model_type", default=None, required=True, type=str, help="Model type selected in the list: "
                                                                                    + ", ".join(MODEL_CLASSES_TRANSFORMERS.keys()))

    # Train and/or Evaluate
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run an evaluation on the test set.")
    parser.add_argument("--do_baseline_eval", action="store_true",
                        help="Whether to run a basline evaluation on the test set.")

    # Logging arguments
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    # CRF related arguments
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignore when calculate loss)")

    # Optional arguments
    parser.add_argument("--slot_label_file", default=SLOT_ANNOTATIONS_FILENAME, type=str, help="Slot Label annotation file")
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
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
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--online", default=True, help="If not connected to the internet, please set this argument to False ")
    parser.add_argument("--use_cuda_device", default=True, type=bool, help="Use GPU device.")

    args = parser.parse_args()
    args.model_dir = os.path.join(MODEL_OUTPUT_DIR, args.model_dir)
    # Direct to offline folder with downloaded pretrained models in case no internet is available
    if args.online:
        args.pretrained_model_path = PRETRAINED_MODEL_PATH[args.model_type.split('_')[1]]
    else:
        args.pretrained_model_path = os.path.join(PRETRAINED_MODELS_DIR, PRETRAINED_MODEL_PATH[args.model_type.split('_')[1]])
    args.do_eval = False if args.do_baseline_eval else args.do_eval
    main(args)
