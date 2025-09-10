import torch
from torch import FloatTensor, nn
from pytorch_lightning import LightningModule


class NTPModel(nn.Module):
    def __init__(
        self,
        token_size: int,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        num_layers: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(token_size, d_model)
        self.transformer_layer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, token_size, bias=False)
        # tie weights
        # self.fc.weight = self.embedding.weight

    def forward(self, x: torch.LongTensor, mask: torch.BoolTensor) -> FloatTensor:
        """Forward

        :param x: input ids of shape [batch_size, seq_len]
        :type x: torch.LongTensor
        :param mask: mask of shape [batch_size, seq_len, seq_len]
        :type mask: torch.BoolTensor
        :returns: logits of shape [batch_size, seq_len, token_size]
        :rtype: FloatTensor
        """
        x = self.embedding(x)  # (B, T, d_model)
        for layer in self.transformer_layer:
            x = layer(x, src_mask=mask)
        x = self.fc(x)  # (B, T, token_size)
        return x


if __name__ == "__main__":
    model = NTPModel(
        token_size=2000,
        d_model=512,
        n_heads=8,
        dim_feedforward=2048,
        num_layers=6,
    )
    x = torch.randint(0, 2000, (2, 128))
    mask = nn.Transformer.generate_square_subsequent_mask(128).bool()
    mask = mask.unsqueeze(0).repeat(2 * 8, 1, 1)  # (B, T, T)
    y = model(x, mask)
    print(y.shape)  # (2, 128, 32000)
