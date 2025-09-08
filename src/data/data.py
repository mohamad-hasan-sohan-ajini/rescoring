from pathlib import Path

import torch
import sentencepiece as spm
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
        ids = [self.bos_index] + ids + [self.eos_index]

        input_ids = self._pad_or_trunc(ids)
        labels = input_ids[1:] + [self.pad_index]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


if __name__ == "__main__":
    dataset = TextLineCausalDataset(
        text_path="dataset/dataset.txt",
        sp_model="tokenizer/unigram_2000.model",
        seq_len=32,
    )
    print(dataset[128])
