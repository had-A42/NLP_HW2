"""A reversible byte-level BPE trained solely on the training partition."""

import hashlib
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers


class CodeTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, path):
        self.path = Path(path)
        self.backend = Tokenizer.from_file(str(self.path))
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = [
            self.backend.token_to_id(token) for token in self.SPECIAL_TOKENS
        ]
        if None in [self.pad_id, self.bos_id, self.eos_id, self.unk_id]:
            raise ValueError("Tokenizer is missing required special tokens.")
        self.vocab_size = self.backend.get_vocab_size()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @classmethod
    def train(cls, texts, path, vocab_size=8192, min_frequency=2):
        if vocab_size < 260:
            raise ValueError(
                "Byte-level BPE needs at least 256 bytes plus four special tokens."
            )
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=True
        )
        backend.decoder = decoders.ByteLevel()
        special = [
            AddedToken(t, special=True, normalized=False) for t in cls.SPECIAL_TOKENS
        ]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        backend.train_from_iterator(texts, trainer=trainer)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        backend.save(str(path))
        return cls(path)

    def encode(self, text):
        # encode_special_tokens makes literal strings like '<eos>' ordinary code text.
        self.backend.encode_special_tokens = True
        return self.backend.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        self.backend.encode_special_tokens = True
        return [
            encoding.ids
            for encoding in self.backend.encode_batch(texts, add_special_tokens=False)
        ]

    def decode(self, ids):
        ids = list(ids)
        if self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        return self.backend.decode(ids, skip_special_tokens=True)
