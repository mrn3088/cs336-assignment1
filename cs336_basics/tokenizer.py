import os
from collections import Counter
from pretokenization import run_pre_tokenization, PreTokenizationRequest

class BPETokenizer:
    """BPE (Byte Pair Encoding) tokenizer implementation."""
    input_path: str
    vocab_size: int
    special_tokens: list[str]

    def __init__(
        self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
    
    def _pre_tokenize(self) -> Counter[bytes]:
        """Pre-tokenize the input file."""
        print(f"Pre-tokenizing {self.input_path}...")
        request = PreTokenizationRequest(file_path=self.input_path)
        return run_pre_tokenization(request)

    def train(self) -> dict[int, bytes]:
        """Train the BPE tokenizer by iterating over the training data."""
        pre_token_counts = self._pre_tokenize()
        print(f"Pre-token number: {len(pre_token_counts)}")
        raise NotImplementedError

    def get_vocab_and_merges(
        self,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Return the vocabulary and merges of the tokenizer.

        Returns:
            A tuple containing:
                - vocab: A mapping from token ID (int) to token bytes (bytes)
                - merges: A list of merge tuples, where each tuple contains two bytes
                  representing tokens that were merged together.
        """
        raise NotImplementedError

