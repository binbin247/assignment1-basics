from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

import regex as re

try:
    from cs336_basics.pretokenization_example import find_chunk_boundaries
except ImportError:
    from pretokenization_example import find_chunk_boundaries

# GPT-2 pre-tokenization pattern from the assignment handout.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)


def _count_pretokens(text: str, special_tokens: list[str]) -> Counter[str]:
    """Split out special tokens, then count GPT-2 pretokens in the remaining text."""
    counts: Counter[str] = Counter()
    if special_tokens:
        special_pat = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        segments = re.split(special_pat, text)
    else:
        segments = [text]

    for segment in segments:
        if not segment:
            continue
        for match in PAT_RE.finditer(segment):
            counts[match.group()] += 1
    return counts


def _merge_word(word: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    """Replace every non-overlapping occurrence of `pair` in `word`, left to right."""
    a, b = pair
    merged = a + b
    new_word: list[bytes] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocab and merge list."""
    input_path = os.fspath(input_path)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"

    word_counts: Counter[str] = Counter()
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks=8, split_special_token=split_special_token)
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            word_counts.update(_count_pretokens(chunk, special_tokens))

    # Represent each unique pretoken as a list of single-byte tokens.
    word_list: list[list[bytes]] = []
    freq_list: list[int] = []
    for word, freq in word_counts.items():
        word_list.append([bytes([b]) for b in word.encode("utf-8")])
        freq_list.append(freq)

    pair_counts: dict[tuple[bytes, bytes], int] = Counter()
    pair_to_indices: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for idx, word in enumerate(word_list):
        freq = freq_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_indices[pair].add(idx)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        if not pair_counts:
            break
        # Highest frequency wins; ties go to the lexicographically greater pair.
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_indices = list(pair_to_indices.get(best_pair, ()))
        for idx in affected_indices:
            word = word_list[idx]
            freq = freq_list[idx]

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                new_count = pair_counts.get(pair, 0) - freq
                if new_count <= 0:
                    pair_counts.pop(pair, None)
                else:
                    pair_counts[pair] = new_count
                indices = pair_to_indices.get(pair)
                if indices is not None:
                    indices.discard(idx)
                    if not indices:
                        pair_to_indices.pop(pair, None)

            new_word = _merge_word(word, best_pair)
            word_list[idx] = new_word

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                pair_to_indices[pair].add(idx)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._bpe_cache: dict[bytes, list[bytes]] = {}

        if self.special_tokens:
            # Longest match first so overlapping special tokens encode correctly.
            special_pat = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile(f"({special_pat})")
            self._special_set = set(self.special_tokens)
        else:
            self._special_re = None
            self._special_set = set()

    def _bpe(self, token: bytes) -> list[bytes]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]

        word = [bytes([b]) for b in token]
        if len(word) < 2:
            self._bpe_cache[token] = word
            return word

        while True:
            best_pair = None
            best_rank = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            word = _merge_word(word, best_pair)

        self._bpe_cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            parts = [text]
        else:
            parts = self._special_re.split(text)

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self._special_set:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue
            for match in PAT_RE.finditer(part):
                for piece in self._bpe(match.group().encode("utf-8")):
                    ids.append(self.bytes_to_id[piece])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    train_data_path = "/data/private/niebinbin/test_and_study/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    val_data_path = "/data/private/niebinbin/test_and_study/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

    vocab, merges = train_bpe(val_data_path, vocab_size=1000, special_tokens=["<|endoftext|>"])
    print("The vocab size is:", len(vocab))
    print("The number of merges is:", len(merges))
    print("The first 10 merges are:", merges[:10])
