import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_layer_size = 80, num_layers = 3, output_size = 1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers = num_layers, bidirectional = True)

        self.linear = nn.Linear(2*hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(2*num_layers, 1, self.hidden_layer_size),
                            torch.zeros(2*num_layers, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
