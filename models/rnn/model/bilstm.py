import torch
import torch.nn as nn
from torch.autograd import Variable
from .module import SlotClassifier
import torch.functional as F


class BiRNN(nn.Module):
    def __init__(self, args, output_size, vocab_size, weights):
        super(BiRNN, self).__init__()
        self.args = args
        self.input_dim = vocab_size
        self.hidden_size = args.hidden_size
        self.output_size = output_size
        self.batch_size = args.train_batch_size
        self.embedding_length = args.max_seq_len

        # Initializing the look-up table
        self.embedding = nn.Embedding(self.input_dim, self.embedding_length)
        if type(weights) == torch.Tensor:
            # Assigning the look-up table to the pre-trained word embedding
            # self.embedding = self.embedding.from_pretrained(weights)
            self.embedding.weight = nn.Parameter(weights, requires_grad=False)

        self.lstm = nn.LSTM(input_size=self.embedding_length, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.slot_classifier = SlotClassifier(self.hidden_size, self.output_size, args.dropout_rate)

    def forward(self, input_sentence, slot_labels_ids):
        # Embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = self.embedding(input_sentence)
        # print(input.size())
        # exit()
        output, _ = self.lstm(input, None)
        slot_logits = self.slot_classifier(torch.squeeze(output, 0))

        total_loss = 0
        # 1. Slot Softmax
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.output_size), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = (total_loss,) + (slot_logits,)
        return outputs