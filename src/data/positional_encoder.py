import math

import torch
from pytorch_lightning import LightningModule


positional_encoding = None


class PositionalEncoding(LightningModule):
    """Sinusoidal Positional Encoding"""

    def __init__(
        self,
        d_model: int,
        max_len: int,
    ):
        """Initialize

        :param d_model: Model hidden size
        :type d_model: int
        :param max_len: maximum length of input signal
        :type max_len: int
        """
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # pe: (T, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the signal

        :param x: input of shape [batch_size, channels (=1), freq, time]
        :type x: torch.Tensor
        :returns: Positional encoded added signal
        :rtype: torch.Tensor
        """
        *_, seq_len = x.shape
        return self.pe[:seq_len]


def get_positional_encoding(d_model: int = 1024, max_len: int = 2000):
    """Get positional encoding module

    :param d_model: Model hidden size
    :type d_model: int
    :param max_len: maximum length of input signal
    :type max_len: int
    :returns: Positional encoding module
    :rtype: PositionalEncoding
    """
    global positional_encoding
    if positional_encoding is None:
        positional_encoding = PositionalEncoding(d_model, max_len)
    return positional_encoding


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pe = get_positional_encoding(1024, 2000)
    plt.figure(figsize=(10, 20))
    plt.imshow(pe.pe)
    plt.savefig("positional_encoding.png")

    # check similarity

    for t in range(0, 2000, 100):
        key = pe.pe[t]
        sim = pe.pe @ key
        plt.clf()
        plt.plot(sim)
        plt.savefig(f"/tmp/sim{t:04d}.png")
