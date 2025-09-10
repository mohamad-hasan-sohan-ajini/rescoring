from pathlib import Path
from typing import Optional

import torch
import sentencepiece as spm
from pytorch_lightning import LightningDataModule
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TextLineCausalDataset(Dataset):
    """
    One-sentence-per-line dataset for next-token prediction (causal LM).
    """

    def __init__(
        self,
        text_path: str,
        sp_model: str,
        seq_len: int,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.lines = Path(text_path).read_text().splitlines()
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)

        self.seq_len = seq_len
        self.alpha = alpha

        # special tokens
        self.pad_index = self.sp.pad_id()
        self.bos_index = self.sp.bos_id()
        self.eos_index = self.sp.eos_id()

    def __len__(self) -> int:
        return len(self.lines)

    def _pad_or_trunc(self, ids: list[int]) -> list[int]:
        if len(ids) > self.seq_len:
            return ids[: self.seq_len]
        elif len(ids) < self.seq_len:
            return ids + [self.pad_index] * (self.seq_len - len(ids))
        return ids

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


def decorate_collate_function(pad_index, max_len):
    def decorate(function):
        def wrap(batch):
            return function(batch, pad_index=pad_index, max_len=max_len)

        return wrap

    return decorate


@decorate_collate_function(pad_index=3, max_len=64)
def collate_function(batch, pad_index, max_len):
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
    return (batch_input_ids, batch_labels, batch_mask)


class ASRDM(LightningDataModule):
    def __init__(
        self,
        train_dataset_path,
        validation_dataset_path,
        sp_model,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.sp_model = sp_model

    def setup(self, stage: Optional[str] = None) -> None:
        tda = TimeDomainAugmentation(
            base_path=self.tda_base_path,
            json_path=self.tda_json_path,
        )
        fda = FrequencyDomainAugmentation()
        self.train_dataset = ASRDS(
            self.train_dataset_path,
            audio_loader=AudioLoader(),
            charset_path=self.charset_path,
            time_domain_augmentor=tda,
            frequency_domain_augmentor=fda,
        )
        self.validation_dataset = ASRDS(
            self.validation_dataset_path,
            audio_loader=AudioLoader(),
            charset_path=self.charset_path,
            time_domain_augmentor=tda,
            frequency_domain_augmentor=fda,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=native_ctc_collate_function,
            pin_memory=True,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            num_workers=2,
            batch_size=self.batch_size,
            collate_fn=native_ctc_collate_function,
            pin_memory=True,
            shuffle=False,
        )


if __name__ == "__main__":
    dataset = TextLineCausalDataset(
        text_path="dataset/dataset.txt",
        sp_model="tokenizer/unigram_2000.model",
        seq_len=16,
    )
    offset = 64
    sample = dataset[offset]
    print(sample)

    batch = [dataset[offset + i] for i in range(4)]
    batch_input_ids, batch_labels, batch_mask = collate_function(batch)
    print(batch_input_ids.shape)
    print(batch_labels.shape)
    print(batch_mask.shape)
