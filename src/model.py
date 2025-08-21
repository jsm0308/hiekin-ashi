import torch
import torch.nn as nn

class BiGRUAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.4):
        super(BiGRUAttentionNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        
        # Attention Mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(out * attn_weights, dim=1)
        
        out = self.dropout(context)
        out = self.fc(out)
        return out