import torch
from torch import FloatTensor, nn
from pytorch_lightning import LightningModule


class NTPModel(LightningModule):
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
        self.fc.weight = self.embedding.weight
        # criterion
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        x: torch.LongTensor,
        pe: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> FloatTensor:
        """Forward

        :param x: input ids of shape [batch_size, seq_len]
        :type x: torch.LongTensor
        :param pe: positional encoding of shape [seq_len, d_model]
        :type pe: torch.FloatTensor
        :param mask: mask of shape [batch_size, seq_len, seq_len]
        :type mask: torch.BoolTensor
        :returns: logits of shape [batch_size, seq_len, token_size]
        :rtype: FloatTensor
        """
        # (B, T, d_model)
        x = self.embedding(x)
        x = x + pe
        for layer in self.transformer_layer:
            x = layer(x, src_mask=mask)
        # (B, T, token_size)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        input_ids, labels, mask, pe = batch
        pred = self(input_ids, pe, mask)
        loss = self.criterion(pred, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids, labels, mask, pe = batch
        pred = self(input_ids, pe, mask)
        loss = self.criterion(pred, labels)
        return {"val_loss": loss}


if __name__ == "__main__":
    model = NTPModel(
        token_size=2000,
        d_model=512,
        n_heads=8,
        dim_feedforward=2048,
        num_layers=6,
    )
    batch_size = 2
    seq_len = 11
    x = torch.randint(0, 2000, (batch_size, seq_len))
    pe = torch.randn(seq_len, 512)
    mask = torch.randint(0, 2, (batch_size * 8, seq_len, seq_len), dtype=torch.bool)
    y = model(x, pe, mask)
    print(y.shape)
