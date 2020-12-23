import os
import logging
import json

from tqdm import tqdm, trange

import numpy as np
import torch

from config.global_config import VAL_FOLDER_NAME, TEST_FOLDER_NAME, CUDA_PREF
from models.rnn.model import BiRNN
from torch.optim import AdamW
from models.baseline.model.static_classifier import StaticClassifier
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from utilities.utils import MODEL_CLASSES_RNN, compute_metrics, get_slot_labels, load_knowledge, create_batch

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, val_dataset=None, test_dataset=None, vocabulary=None, pretrained_weights=None):
        assert vocabulary
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.vocabulary = vocabulary
        self.pretrained_weights = pretrained_weights
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        # Use CPU if explicitly mentioned otherwise use default or specified GPU device
        self.device = "cpu" if args.no_cuda else "cuda:"+str(CUDA_PREF) if torch.cuda.is_available and args.use_cuda_device else "cuda"
        self.model = self.determine_model_type(args)

        # Sent model to device
        print(self.model)
        print("Using GPU Device => {}".format(self.device))
        self.model.to(self.device)

    def determine_model_type(self, args):
        if args.model_type in MODEL_CLASSES_RNN.keys():
            return MODEL_CLASSES_RNN[args.model_type](args=args,
                                                      output_size=len(self.slot_label_lst),
                                                      vocab_size=len(self.vocabulary['word2index'].keys()),
                                                      weights=self.pretrained_weights)
        else:
            raise Exception("The provided model is not known")

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_sentence': batch[0],
                          'slot_labels_ids': batch[1]}

                # Change this when implementing different models
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                # Gradient clipping to prevent exploding gradients
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()  # Update parameters
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate_neural(VAL_FOLDER_NAME, step=global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate_baseline(self, mode, args):
        """
        Evaluation based on static models
        :return:
        """
        # Load knowledge and intialize static model
        knowledge = load_knowledge()
        slot_labels = get_slot_labels(args)
        static_model = StaticClassifier(knowledge=knowledge, slot_labels=slot_labels)

        # Eval!
        y_pred_list = []
        y_true_list = []
        logger.info("***** Running evaluation on %s dataset *****", mode)
        with open(os.path.join(args.data_dir, args.task, mode, 'seq.in'), 'r') as seq_in, open(os.path.join(args.data_dir, args.task, mode, 'seq.out'), 'r') as seq_out:
            for l_in, l_out in zip(seq_in, seq_out):
                l_in = str(l_in).rstrip()
                l_out = str(l_out).rstrip()

                pred = static_model.predict(l_in)
                y_true_list.append(l_out.split(' '))
                y_pred_list.append(pred)

        # Compute metrics
        results = {}
        total_result = compute_metrics(y_pred_list, y_true_list)
        results.update(total_result)
        logger.info("***** Eval results *****")
        for key, value in results.items():
            if key == 'label_metrics':
                logger.info("--- label metrics ---")
                for label, label_value in value.items():
                    logger.info("  %s = %s", label, str(label_value))
            else:
                logger.info("  %s = %s", key, str(value))
        # self.save_eval(results, mode)
        return results

    def evaluate_neural(self, mode, step=None):
        """
        Evaluation based on neural models
        :param mode:
        :param step:
        :return:
        """
        if mode == TEST_FOLDER_NAME:
            dataset = self.test_dataset
        elif mode == VAL_FOLDER_NAME:
            dataset = self.val_dataset
        else:
            raise Exception("Only val and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        slot_preds = None
        out_slot_labels_ids = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_sentence': batch[0],
                          'slot_labels_ids': batch[1]}

                # Change this when implementing different models
                outputs = self.model(**inputs)
                tmp_eval_loss, slot_logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            # Slot prediction
            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(slot_preds_list, out_slot_label_list)
        results.update(total_result)
        if step:
            results.update({'epoch': step})

        logger.info("***** Eval results *****")
        for key, value in results.items():
            if key == 'label_metrics':
                logger.info("--- label metrics ---")
                for label, label_value in value.items():
                    logger.info("  %s = %s", label, str(label_value))
            else:
                logger.info("  %s = %s", key, str(value))
        self.save_eval(results, mode)
        return results

    def save_eval(self, results, mode):
        print('--- SAVING (INTERMEDIATE) EVAL RESULTS ---')
        if not os.path.exists(os.path.join(self.args.model_dir, 'eval')):
            os.makedirs(os.path.join(self.args.model_dir, 'eval'))

        try:
            if mode == VAL_FOLDER_NAME:
                # Create file if not exists
                if not os.path.exists(os.path.join(self.args.model_dir, 'eval', 'intermediate_evaluation.json')):
                    open(os.path.join(self.args.model_dir, 'eval', 'intermediate_evaluation.json'), 'w').close()

                # Append results to new file
                with open(os.path.join(self.args.model_dir, 'eval', 'intermediate_evaluation.json'), 'a') as f:
                    json.dump(results, f)
            else:
                # Create or overwrite
                with open(os.path.join(self.args.model_dir, 'eval', 'test_evaluation.json'), 'w') as f:
                    json.dump(results, f)
            logger.info("***** Evaluation results saved *****")
        except FileExistsError:
            raise Exception("Cannot write evaluation results to file")

    def save_model(self):
        print('--- SAVING MODEL ---')
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = MODEL_CLASSES_RNN[self.args.model_type](args=self.args,
                                                      output_size=len(self.slot_label_lst),
                                                      vocab_size=len(self.vocabulary['word2index'].keys()),
                                                      weights=self.pretrained_weights)
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'training_args.bin')))
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except FileNotFoundError:
            raise Exception("Some model files might be missing...")
