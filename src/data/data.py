from pathlib import Path
from typing import Optional

import torch
import sentencepiece as spm
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .positional_encoder import get_positional_encoding

positional_encoding = get_positional_encoding(d_model=512, max_len=1200)


class NTPDataset(Dataset):
    """
    One-sentence-per-line dataset for next-token prediction (causal LM).
    """

    def __init__(
        self,
        text_path: str,
        sp_model_path: str,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.lines = Path(text_path).read_text().splitlines()
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        self.alpha = alpha

        # special tokens
        self.pad_index = self.sp.pad_id()
        self.bos_index = self.sp.bos_id()
        self.eos_index = self.sp.eos_id()

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.lines[idx]
        ids = self.sp.sample_encode_as_ids(text, nbest_size=-1, alpha=self.alpha)
        input_ids = [self.bos_index] + ids
        labels = ids + [self.eos_index]
        length = len(ids) + 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": length,
        }


def decorate_collate_function(pad_index, max_len, n_heads):
    def decorate(function):
        def wrap(batch):
            return function(
                batch,
                pad_index=pad_index,
                max_len=max_len,
                n_heads=n_heads,
            )

        return wrap

    return decorate


@decorate_collate_function(pad_index=3, max_len=1024, n_heads=8)
def collate_function(batch, pad_index, max_len, n_heads):
    batch_size = len(batch)
    batch_len = max([sample["length"] for sample in batch])

    batch_input_ids = torch.LongTensor(batch_size, batch_len).fill_(pad_index)
    batch_labels = torch.LongTensor(batch_size, batch_len).fill_(pad_index)
    batch_mask = torch.BoolTensor(batch_size, batch_len, batch_len).fill_(True)

    for i, sample in enumerate(batch):
        sample_length = sample["length"]
        # input
        batch_input_ids[i, :sample_length] = torch.LongTensor(sample["input_ids"])
        # labels
        batch_labels[i, :sample_length] = torch.LongTensor(sample["labels"])
        # pad mask
        pad_mask = torch.BoolTensor(batch_len).fill_(True)
        pad_mask[:sample_length].fill_(False)
        # causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(batch_len)
        causal_mask = causal_mask.bool()
        # aggregate masks
        mask = causal_mask | pad_mask.unsqueeze(0) | pad_mask.unsqueeze(1)
        batch_mask[i] = mask
    # trim to max sequence length
    batch_input_ids = batch_input_ids[:, :max_len]
    batch_labels = batch_labels[:, :max_len]
    batch_mask = batch_mask[:, :max_len, :max_len]
    batch_len = min(batch_len, max_len)
    batch_mask = (
        batch_mask.view(batch_size, 1, batch_len, batch_len)
        .expand(-1, n_heads, -1, -1)
        .clone()
        .view(batch_size * n_heads, batch_len, batch_len)
    )
    # positional encoding
    pe = positional_encoding.pe[:batch_len]
    return (
        batch_input_ids.contiguous(),
        batch_labels.contiguous(),
        batch_mask.contiguous(),
        pe.contiguous(),
    )


class NTPDM(LightningDataModule):
    def __init__(
        self,
        train_dataset_path: str,
        validation_dataset_path: str,
        sp_model_path: str,
        num_workers: int,
        batch_size: int,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.sp_model_path = sp_model_path
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NTPDataset(
            self.train_dataset_path,
            self.sp_model_path,
        )
        self.validation_dataset = NTPDataset(
            self.validation_dataset_path,
            self.sp_model_path,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_function,
            pin_memory=True,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_function,
            pin_memory=True,
            shuffle=False,
        )


if __name__ == "__main__":
    print("TEST DATASET")
    dataset = NTPDataset(
        text_path="dataset/dataset.txt",
        sp_model_path="tokenizer/unigram_2000.model",
    )
    offset = 64
    sample = dataset[offset]
    print(sample)

    batch = [dataset[offset + i] for i in range(4)]
    batch_input_ids, batch_labels, batch_mask, pe = collate_function(batch)
    print(batch_input_ids.shape)
    print(batch_labels.shape)
    print(batch_mask.shape)

    # torchlightning datamodule
    print("TEST DATAMODULE")
    datamodule = NTPDM(
        train_dataset_path="dataset/dataset.txt",
        validation_dataset_path="dataset/dataset.txt",
        sp_model_path="tokenizer/unigram_2000.model",
        num_workers=2,
        batch_size=4,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    for batch_input_ids, batch_labels, batch_mask, pe in train_loader:
        print(batch_input_ids.shape)
        print(batch_labels.shape)
        print(batch_mask.shape)
        print(pe.shape)
        break
