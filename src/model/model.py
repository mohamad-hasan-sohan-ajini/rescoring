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
        self.token_size = token_size
        self.embedding = nn.Embedding(token_size, d_model)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    # activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, token_size, bias=False)
        # tie weights
        self.fc.weight = self.embedding.weight
        # criterion (ignore padding index in the loss)
        self.criterion = nn.NLLLoss(ignore_index=3)

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
        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)
        # (B, T, token_size)
        x = self.fc(x).log_softmax(2)
        return x

    def training_step(self, batch, batch_idx):
        input_ids, labels, mask, pe = batch
        batch_size = input_ids.size(0)
        pred = self(input_ids, pe, mask).view(-1, self.token_size)
        loss = self.criterion(pred, labels.view(-1))
        self.log("train_loss", loss.item(), batch_size=batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids, labels, mask, pe = batch
        batch_size = input_ids.size(0)
        pred = self(input_ids, pe, mask).view(-1, self.token_size)
        loss = self.criterion(pred, labels.view(-1))
        self.log("val_loss", loss.item(), batch_size=batch_size)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.embedding.parameters(), "lr": 3e-5},
                {"params": self.transformer_layers.parameters(), "lr": 1e-5},
                # {"params": self.fc.parameters(), "lr": 1e-3},  # comment if weights are tied
            ],
        )
        # Use scheduler for periodic lr increase/decrease
        # multiply `step_size[up|down]` by `accumulate_grad_batches` to find number of training steps in each slop!
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            max_lr=3e-4,
            base_lr=1e-5,
            step_size_down=100_000,
            step_size_up=100_000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


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
