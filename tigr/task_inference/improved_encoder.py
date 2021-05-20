import torch
import torch.nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
import numpy as np


class GruAttentionEncoder(nn.Module):
    def __init__(self, input_size, output_size, time_steps, encoding_mode):
        super().__init__()
        # Note: encoding mode is automatically set to trajectory
        self.input_size = input_size // time_steps
        self.output_size = output_size
        self.time_steps = time_steps
        self.encoding_mode = encoding_mode

        self.gru = nn.GRU(self.input_size, self.output_size, batch_first=True)
        self.attn = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        hidden = self.initHidden(x.size(0))

        x = x.view(x.size(0), self.time_steps, self.input_size)
        _, hidden = self.gru(x, hidden)

        hidden = hidden.squeeze(0)

        # TODO: Test
        # attn_weights = F.softmax(self.attn(hidden), dim=-1)
        # hidden = attn_weights * hidden

        hidden = F.relu(hidden)
        return hidden

    def initHidden(self, b_size):
        return ptu.zeros(1, b_size, self.output_size)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvAttentionEncoder(nn.Module):
    def __init__(self, input_size, output_size, time_steps, encoding_mode):
        super().__init__()
        # Note: encoding mode is automatically set to transitionSharedY
        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.encoding_mode = encoding_mode

        self.conv_layer = torch.nn.Sequential(
            nn.Conv1d(1, self.output_size, self.input_size, self.input_size),
            nn.ReLU(),
            nn.Conv1d(self.output_size, self.output_size, 1, 1),
            nn.ReLU()
        )

        self.attn = nn.Conv1d(self.output_size, 1, 1, 1)

        self.extraction_layers = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        # Bring x to correct shape for convs (needed for trajectory and sharedy!)
        x = x.view(x.size(0), 1, self.input_size * self.time_steps)

        x = self.conv_layer(x)

        # TODO: Test
        # # Calculate attention weights per timestep and apply
        # attn_weights = F.softmax(self.attn(x), dim=-1)
        # x = x * attn_weights

        # Flatten for linear layers and apply rest
        x = x.permute(0, 2, 1)
        x = self.extraction_layers(x)

        return x


class TransformerModel(nn.Module):

    def __init__(self, input_size, output_size, time_steps, encoding_mode, nhead=1, nlayers=2, dropout=0.5):
        super(TransformerModel, self).__init__()
        # Note: encoding mode is automatically set to transitionSharedY!
        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.encoding_mode = encoding_mode

        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout

        self.pos_encoder = PositionalEncoding(self.input_size, self.dropout)

        encoder_layers = nn.TransformerEncoderLayer(self.input_size, self.nhead, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)

        self.extraction_layers = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # Prepare batch for transformer
        x = x.permute(1, 0, 2)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Reorder batch for GMM
        x = x.permute(1, 0, 2)

        # Bring to shared dim size
        x = self.extraction_layers(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, input_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.input_size)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_size, 2).float() * (-np.log(10000.0) / self.input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if self.input_size % 2 == 0 else torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
