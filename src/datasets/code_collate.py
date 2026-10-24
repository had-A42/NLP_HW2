import torch
from torch.nn.utils.rnn import pad_sequence


class CodeCollator:
    def __init__(self, pad_id=0, bos_id=1):
        self.pad_id, self.bos_id = pad_id, bos_id

    def __call__(self, items):
        source = [torch.tensor(item["source_ids"], dtype=torch.long) for item in items]
        input_ids = pad_sequence(source, batch_first=True, padding_value=self.pad_id)
        batch = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_id),
            "row_id": torch.tensor(
                [item["row_id"] for item in items], dtype=torch.long
            ),
        }
        if "target_ids" in items[0]:
            targets = [
                torch.tensor(item["target_ids"], dtype=torch.long) for item in items
            ]
            labels = pad_sequence(targets, batch_first=True, padding_value=-100)
            decoder = torch.full_like(labels, self.pad_id)
            decoder[:, 0] = self.bos_id
            decoder[:, 1:] = labels[:, :-1].masked_fill(
                labels[:, :-1].eq(-100), self.pad_id
            )
            batch.update(labels=labels, decoder_input_ids=decoder)
        return batch
