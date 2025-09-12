import json

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.data import NTPDM
from model.model import NTPModel

# create dataset
datamodule = NTPDM(
    train_dataset_path="/home/aj/baden/lm-data/aggregation_2025-09-06.txt",
    validation_dataset_path="/home/aj/baden/lm-data/test.txt",
    sp_model_path="/home/aj/repo/rescoring/src/data/tokenizer/unigram_2000.model",
    num_workers=8,
    batch_size=32,
)
datamodule.setup()

# create model
model = NTPModel(
    token_size=2000,
    d_model=512,
    n_heads=8,
    dim_feedforward=2048,
    num_layers=6,
)

# checkpointing
loss_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="val_loss",
    mode="min",
    save_last=True,
    every_n_train_steps=1_000,
    dirpath="/data/checkpoints",
    filename="minloss-{epoch}-{step}",
)

trainer = Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=1000,
    callbacks=[loss_callback],
    # accumulate_grad_batches=1,
    # precision="32",
    # gradient_clip_val=0.1,
    val_check_interval=10_000,
)
trainer.fit(model, datamodule)
