import torch
from torch.nn.utils.rnn import pad_sequence

class UnlearningDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _collate(self, features):
        return {
            "input_ids": pad_sequence(
                [torch.tensor(f["input_ids"]) for f in features],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [torch.tensor(f["attention_mask"]) for f in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [torch.tensor(f["labels"]) for f in features],
                batch_first=True,
                padding_value=-100,
            ),
        }

    def __call__(self, batch):
        return {
            "forget": self._collate([x["forget"] for x in batch]),
            "retain": self._collate([x["retain"] for x in batch]),
        }