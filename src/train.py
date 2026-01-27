from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.data import NTPDM
from model.model import NTPModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

STEPS = 100_000

# create dataset
base_path = Path(__file__).parent
dataset_path = base_path / "data" / "dataset"
datamodule = NTPDM(
    train_dataset_path=dataset_path / "train.txt",
    validation_dataset_path=dataset_path / "validation.txt",
    sp_model_path=str(base_path / "data" / "tokenizer" / "unigram_2000.model"),
    num_workers=8,
    batch_size=8,
)
datamodule.setup()

# create model
model = NTPModel(
    token_size=2000,
    d_model=512,
    num_layers=6,
)

# checkpointing
loss_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="val_loss",
    mode="min",
    save_last=True,
    every_n_train_steps=STEPS,
    dirpath="checkpoints",
    filename="minloss-{epoch}-{step}",
)
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=1000,
    callbacks=[loss_callback, lr_monitor],
    accumulate_grad_batches=4,
    precision="16-mixed",
    # gradient_clip_val=0.1,
    val_check_interval=STEPS,
)
trainer.fit(model, datamodule)
