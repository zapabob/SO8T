from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class Vocabulary:
    """Lightweight whitespace vocabulary with JSON persistence."""

    def __init__(self, tokens: Sequence[str] | None = None):
        specials = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self._itos: List[str] = list(dict.fromkeys(specials))
        self._stoi: Dict[str, int] = {token: idx for idx, token in enumerate(self._itos)}
        if tokens:
            for token in tokens:
                self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self._stoi:
            self._stoi[token] = len(self._itos)
            self._itos.append(token)
        return self._stoi[token]

    def build_from_iterator(self, iterator: Iterable[Sequence[str]], min_freq: int = 1) -> None:
        counter: Counter[str] = Counter()
        for tokens in iterator:
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= min_freq:
                self.add_token(token)

    def encode(self, tokens: Sequence[str], include_special_tokens: bool = True) -> List[int]:
        indices: List[int] = []
        if include_special_tokens:
            indices.append(self[BOS_TOKEN])
        for token in tokens:
            indices.append(self[token])
        if include_special_tokens:
            indices.append(self[EOS_TOKEN])
        return indices

    def decode(self, indices: Sequence[int], skip_special_tokens: bool = True) -> List[str]:
        tokens: List[str] = []
        for index in indices:
            token = self._itos[index]
            if skip_special_tokens and token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}:
                continue
            tokens.append(token)
        return tokens

    def __contains__(self, item: str) -> bool:
        return item in self._stoi

    def __getitem__(self, token: str) -> int:
        return self._stoi.get(token, self._stoi[UNK_TOKEN])

    def __len__(self) -> int:
        return len(self._itos)

    @property
    def pad_index(self) -> int:
        return self[PAD_TOKEN]

    def to_dict(self) -> Dict[str, object]:
        return {"itos": self._itos}

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(tokens=data["itos"])
        return vocab

    def to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
