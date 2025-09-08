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
        self.sos_index = self.sp.sos_id()
        self.eos_index = self.sp.eos_id()

    def __len__(self) -> int:
        return len(self.lines)
