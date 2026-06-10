import torch.nn as nn

class TrafficLSTM(nn.Module):
    """
    Two layer LSTM followed by 2 fully-connected layers for single-step
    traffic volume forecasting.

    Input (48 features per sensor)
        -> LSTM (2 layers, hidden_size = 64, dropout = 0.2)
        -> Linear(64 -> 32) + ReLU
        -> Linear(32 -> 1) -- predicted normalized traffic volume
    """
    def __init__(self, input_size = 48, hidden_size = 64, num_layers = 2, dropout = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True, dropout = dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32),
                                  nn.ReLU(),nn.Linear(32, 1))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.head(lstm_out[:, -1, :])
        return out.squeeze(-1)