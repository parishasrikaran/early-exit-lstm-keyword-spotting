"""Causal LSTM with optional running (mean, variation) feature summaries.

The running summaries are calculated from observed audio only, so they preserve
streaming causality. The state includes LSTM state and accumulated stats; it can
be passed from one chunk to the next. Both comparison models share this encoder.
"""
from pathlib import Path
import torch
from torch import nn
from config import N_MELS, NUM_CLASSES


class KeywordLSTM(nn.Module):
    def __init__(self, hidden=128, layers=1, running_stats=True):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.running_stats = running_stats
        self.rnn = nn.LSTM(N_MELS, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden + (2 * N_MELS if running_stats else 0), NUM_CLASSES)
        self.register_buffer('feature_mean', torch.zeros(N_MELS))
        self.register_buffer('feature_std', torch.ones(N_MELS))

    def set_normalization(self, mean, std):
        self.feature_mean.copy_(mean.detach().to(self.feature_mean.device))
        self.feature_std.copy_(std.detach().clamp_min(1e-4).to(self.feature_std.device))

    def forward(self, features, state=None):
        x = (features - self.feature_mean) / self.feature_std
        if self.running_stats:
            if state is None:
                lstm_state = None
                prev_sum = x.new_zeros(x.shape[0], N_MELS)
                prev_sq = x.new_zeros(x.shape[0], N_MELS)
                prev_n = x.new_zeros(x.shape[0], 1)
            else:
                lstm_state, prev_sum, prev_sq, prev_n = state
            h, new_lstm = self.rnn(x, lstm_state)
            sum_ = x.cumsum(dim=1) + prev_sum[:, None, :]
            sq = x.square().cumsum(dim=1) + prev_sq[:, None, :]
            n = torch.arange(1, x.shape[1]+1, device=x.device, dtype=x.dtype)[None, :, None] + prev_n[:, None, :]
            mean = sum_ / n
            std = (sq / n - mean.square()).clamp_min(1e-6).sqrt()
            logits = self.head(torch.cat([h, mean, std], dim=-1))
            new_state = (new_lstm, sum_[:, -1, :], sq[:, -1, :], n[:, -1, :])
        else:
            if state is None:
                lstm_state = None
            else:
                lstm_state = state
            h, new_state = self.rnn(x, lstm_state)
            logits = self.head(h)
        return logits, new_state


def save_checkpoint(path, model, epoch, mode):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'hidden': model.hidden, 'layers': model.layers,
        'running_stats': model.running_stats,
        'epoch': epoch, 'mode': mode,
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = KeywordLSTM(ckpt['hidden'], ckpt['layers'], ckpt.get('running_stats', False)).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    return model, ckpt
