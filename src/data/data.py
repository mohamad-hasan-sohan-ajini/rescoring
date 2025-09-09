from pathlib import Path

import torch
import sentencepiece as spm
from torch import nn
from torch.utils.data import Dataset


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

        # causal mask for token prediction
        # pad_index_mask = [1 if tok != self.pad_index else 0 for tok in input_ids]
        # pad_index_mask = torch.tensor(pad_index_mask, dtype=torch.bool)
        # tril = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))
        # causal_mask = tril & pad_index_mask.unsqueeze(0) & pad_index_mask.unsqueeze(1)

        return {
            "input_ids": input_ids,
            "labels": labels,
            # "causal_mask": causal_mask,
            "length": length,
        }


def collate_function(batch, pad_index: int = 3):
    batch_size = len(batch)
    max_len = max([sample["length"] for sample in batch])

    batch_input_ids = torch.LongTensor(batch_size, max_len).fill_(pad_index)
    batch_labels = torch.LongTensor(batch_size, max_len).fill_(pad_index)
    batch_mask = torch.BoolTensor(batch_size, max_len, max_len).fill_(True)

    for i, sample in enumerate(batch):
        sample_length = sample["length"]
        # input
        batch_input_ids[i, :sample_length] = torch.LongTensor(sample["input_ids"])
        # labels
        batch_labels[i, :sample_length] = torch.LongTensor(sample["labels"])
        # pad mask
        pad_mask = torch.BoolTensor(max_len).fill_(True)
        pad_mask[:sample_length].fill_(False)
        # causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        causal_mask = causal_mask.bool()
        # aggregate masks
        mask = causal_mask | pad_mask.unsqueeze(0) | pad_mask.unsqueeze(1)
        batch_mask[i] = mask
    return (batch_input_ids, batch_labels, batch_mask)


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
