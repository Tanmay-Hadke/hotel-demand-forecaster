# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_scores = self.attention_weights(lstm_out) 
        attn_weights = F.softmax(attn_scores, dim=1) 
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc(context_vector)
        return out