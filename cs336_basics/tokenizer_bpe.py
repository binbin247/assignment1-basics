import os
from pretokenization_example import find_chunk_boundaries
import regex as re
from collections.abc import Iterable
from regex import Match

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. 
    Step:
    1. Pre-tokenize the data into words using a simple regex
    """

    desired_num_chunks = 8
    merges = []

    # vocabulary initialization (256 bytes + special tokens)
    vocab = {i:bytes([i]) for i in range(256)}
    vocab.update({(i+256):special_token.encode('utf-8') for i, special_token in enumerate(special_tokens)})

    # Find the chunk boundaries
    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token = b"<|endoftext|>")
        print("The chunk boundaries are: ", chunk_boundaries)

    # Construc counts: dict [word, count]
    word_counts = {}
    with open(input_path, 'rb') as f:
        iter_pre_tokenization = pre_tokenization(f.read(chunk_boundaries[1]-chunk_boundaries[0]).decode('utf-8'))
        for match in iter_pre_tokenization:
            token = match.group()
            word_counts[token] = word_counts.get(token, 0) + 1

    # Construct bp_counts: dict [bp, count]
    bp_counts = {}
    for key, value in word_counts.items():
        key.encode('utf-8')
        for i in range(len(key)-1):
            bp = key[i:i+2]
            bp_counts[bp] = bp_counts.get(bp,0) + value

    print("The bp_counts are: ", bp_counts['oe'])

    print("The vocab size is: ", len(vocab))
    return vocab, merges


def pre_tokenization(input_text: str) -> Iterable[Match[str]]:
    """
    Pre-tokenize the data into words using a simple regex
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.finditer(PAT, input_text)

if __name__ == "__main__":
    train_data_path = '/data/private/niebinbin/test_and_study/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    val_data_path = '/data/private/niebinbin/test_and_study/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt'

    # with open(val_data_path, 'rb') as f:
    #     # val_data = f.read()
    #     # print("The first 100 characters of the validation data are:\n ", val_data[:100].decode('utf-8'))
    #     # print("The last 100 characters of the validation data are:\n ", val_data[-100:])

    vocab, merges = train_bpe(val_data_path, vocab_size = 1000, special_tokens = ["<|endoftext|>"])
    # print("The vocabulary is: ", vocab)
    # print("The merges are: ", merges)


