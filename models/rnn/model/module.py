import torch.nn as nn


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim * 2, num_slot_labels) # input * 2 for bidirectional

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
