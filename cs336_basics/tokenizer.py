import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


from .pretokenization import run_pre_tokenization, PreTokenizationRequest, pre_tokenize_text_iter


@dataclass
class PreTokenInfo:
    """Record the current bytes sequence of a pre token"""

    mc_id: int
    bytes_sequence: list[bytes]
    token_freq: int
    byte_pair_freq: Counter[tuple[bytes, bytes]] = field(init=False)
    mc_controller: "MergeController" = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Automatically calculate initial byte pair frequencies"""
        self.byte_pair_freq = Counter()
        for b1, b2 in itertools.pairwise(self.bytes_sequence):
            # Here we need to add token_freq instead of 1
            self.byte_pair_freq[(b1, b2)] += self.token_freq

    @classmethod
    def from_pre_token_count(cls, mc_id: int, pre_token_bytes: bytes, count: int) -> "PreTokenInfo":
        """Convert a byte sequence + count pair to PreTokenInfo object"""
        bytes_sequence = [bytes([b]) for b in pre_token_bytes]
        return cls(mc_id=mc_id, bytes_sequence=bytes_sequence, token_freq=count)

    def merge(self, pair: tuple[bytes, bytes]) -> None:
        """
        Merge two bytes `b1` and `b2`
        Example:
        >>> pre_token_info = PreTokenInfo.from_pre_token_count(b"hello", 1)
        >>> pre_token_info.byte_pair_freq
        Counter({(b"h", b"e"): 1, (b"e", b"l"): 1, (b"l", b"l"): 1, (b"l", b"o"): 1})
        >>> pre_token_info.bytes_sequence
        [b"h", b"e", b"l", b"l", b"o"]
        >>> pre_token_info.merge((b"l", b"o"))
        >>> pre_token_info.byte_pair_freq
        Counter({(b"h", b"e"): 1, (b"e", b"l"): 1, (b"l", b"lo"): 1})
        >>> pre_token_info.bytes_sequence
        [b"h", b"e", b"l", b"lo"]
        """
        assert pair in self.byte_pair_freq, f"Pair {pair} not found in byte pair frequency"
        # Update bytes sequence by merging the pair
        new_bytes_sequence = []
        i = 0
        while i < len(self.bytes_sequence):
            if (
                i < len(self.bytes_sequence) - 1
                and self.bytes_sequence[i] == pair[0]
                and self.bytes_sequence[i + 1] == pair[1]
            ):
                new_bytes_sequence.append(pair[0] + pair[1])
                i += 2
            else:
                new_bytes_sequence.append(self.bytes_sequence[i])
                i += 1
        self.bytes_sequence = new_bytes_sequence
        # Delete the byte pair from the merge controller
        old_pairs = list(self.byte_pair_freq.keys())
        for p in old_pairs:
            subs = self.mc_controller.byte_pair_to_subscribers[p]
            subs.discard(self.mc_id)
            if not subs:
                del self.mc_controller.byte_pair_to_subscribers[p]
        self.mc_controller.global_byte_pair_freq.subtract(self.byte_pair_freq)
        # Update byte pair frequency
        self.byte_pair_freq = Counter()
        for b1, b2 in itertools.pairwise(self.bytes_sequence):
            self.byte_pair_freq[(b1, b2)] += self.token_freq
            # Add the new byte pair to the merge controller
            self.mc_controller.byte_pair_to_subscribers[(
                b1, b2)].add(self.mc_id)
        self.mc_controller.global_byte_pair_freq.update(self.byte_pair_freq)

    def subscribe(self, mc_controller: "MergeController") -> None:
        """Register self to all byte pairs in the merge controller"""
        for byte_pair in self.byte_pair_freq.keys():
            mc_controller.byte_pair_to_subscribers[byte_pair].add(self.mc_id)
        mc_controller.pre_token_infos[self.mc_id] = self
        mc_controller.global_byte_pair_freq.update(self.byte_pair_freq)
        self.mc_controller = mc_controller


@dataclass
class MergeController:
    """Controller for merging byte pairs"""

    pre_token_infos: dict[int, PreTokenInfo] = field(default_factory=dict)
    byte_pair_to_subscribers: defaultdict[tuple[bytes, bytes], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    global_byte_pair_freq: Counter[tuple[bytes, bytes]] = field(
        default_factory=Counter)

    def merge(self, byte_pair: tuple[bytes, bytes]) -> None:
        """
        Merge the given byte pair.
        """
        assert byte_pair in self.byte_pair_to_subscribers, f"Byte pair {byte_pair} not found in subscribers"
        for pre_token_info_id in list(
            self.byte_pair_to_subscribers[byte_pair]
        ):  # Make a copy of the set to avoid modifying it while iterating
            self.pre_token_infos[pre_token_info_id].merge(byte_pair)

    def get_most_frequent_byte_pair(self) -> tuple[bytes, bytes] | None:
        """
        Get the most frequent byte pair in the global byte pair frequency.
        To break ties, we use the lexicographical order of the byte pair.
        """
        items = [(p, c)
                 for p, c in self.global_byte_pair_freq.items() if c > 0]
        if not items:
            return None
        return max(items, key=lambda x: (x[1], x[0]))[0]


class BPETokenizer:
    """BPE (Byte Pair Encoding) tokenizer implementation."""

    vocab_size: int
    special_tokens: list[str]
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    vocab_reverse: dict[bytes, int]
    merge_rank: dict[tuple[bytes, bytes], int]

    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.vocab = {}
        self.merges = []
        self.merge_rank = {}
        self.vocab_reverse = {}
        # Add special tokens
        for i, special_token in enumerate(special_tokens):
            b = special_token.encode("utf-8")
            self.vocab[i] = b
            self.vocab_reverse[b] = i
        # Add single byte tokens
        num_special_token = len(self.vocab)
        for i in range(256):
            b = bytes([i])
            self.vocab[i + num_special_token] = b
            self.vocab_reverse[b] = i + num_special_token

    @classmethod
    def from_trained_vocab_and_merges(
        cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        """
        Create a BPE tokenizer from a trained vocabulary and merges.
        Args:
            vocab: A mapping from token ID (int) to token bytes (bytes)
            merges: A list of merge tuples, where each tuple contains two bytes
                representing tokens that were merged together.
            special_tokens: A list of special tokens to add to the tokenizer.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """
        tokenizer = cls(vocab_size=len(vocab),
                        special_tokens=special_tokens or [])
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.vocab_reverse = {b: i for i, b in vocab.items()}
        tokenizer._build_merge_rank()
        return tokenizer

    def _pre_tokenize(self, input_path: str) -> Counter[bytes]:
        """Pre-tokenize the input file."""
        print(f"Pre-tokenizing {input_path}...")
        request = PreTokenizationRequest(
            file_path=input_path, special_tokens=self.special_tokens)
        return run_pre_tokenization(request)

    def _build_merge_rank(self) -> None:
        """Build the merge rank dictionary."""
        merge_rank = {m: i for i, m in enumerate(self.merges)}
        self.merge_rank = merge_rank

    def train(self, input_path: str) -> None:
        """Train the BPE tokenizer by iterating over the training data."""
        pre_token_counts = self._pre_tokenize(input_path)
        # Create the merge controller and subscribe the pre-token infos
        next_available_id = len(self.vocab)
        mc_controller = MergeController()
        for i, (pre_token_id, pre_token_count) in enumerate(pre_token_counts.items()):
            pre_token_info = PreTokenInfo.from_pre_token_count(
                i, pre_token_id, pre_token_count)
            pre_token_info.subscribe(mc_controller)
        # Main training loop
        while len(self.vocab) < self.vocab_size:
            most_frequent_byte_pair = mc_controller.get_most_frequent_byte_pair()
            if most_frequent_byte_pair is None:
                break
            self.merges.append(most_frequent_byte_pair)
            mc_controller.merge(most_frequent_byte_pair)
            new_token = most_frequent_byte_pair[0] + most_frequent_byte_pair[1]
            self.vocab[next_available_id] = new_token
            self.vocab_reverse[new_token] = next_available_id
            next_available_id += 1
        # Build the merge rank dictionary
        self._build_merge_rank()

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
        return self.vocab, self.merges

    def _find_token_to_merge(self, tokens: list[bytes]) -> tuple[list[bytes], bool]:
        """
        If there's a token in the list that can be merged with the next token, merge it, and return the merged token list.
        """
        best: tuple[int, int] | None = None  # (rank, i)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair not in self.merge_rank:
                continue
            rank = self.merge_rank[pair]
            if best is None or rank < best[0]:
                best = (rank, i)
        if best is None:
            return tokens, False
        else:
            return tokens[: best[1]] + [tokens[best[1]] + tokens[best[1] + 1]] + tokens[best[1] + 2:], True

    def _encode_token(self, token: bytes) -> list[int]:
        """Encode a token based on merges and vocab"""
        toks = [bytes([b]) for b in token]
        while True:
            toks, can_merge = self._find_token_to_merge(toks)
            if not can_merge:
                break
        return [self.vocab_reverse[tok] for tok in toks]

    def _encode_part(self, part: str) -> list[int]:
        """
        Encode a part (without special tokens) of text into a list of token IDs.
        """
        ids = []
        for token in pre_tokenize_text_iter(part):
            ids.extend(self._encode_token(token))
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode a text string into a list of token IDs.
        """
        if not text:
            return []
        # Split text by special tokens
        special_pattern = "(" + "|".join(re.escape(s)
                                         for s in self.special_tokens) + ")"
        parts = re.split(special_pattern, text)
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.vocab_reverse[part.encode("utf-8")])
            else:
                ids.extend(self._encode_part(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization
        of large files that we cannot directly load into memory.
        """
        # Build the special token pattern once (same as in encode)
        special_pattern = "(" + "|".join(re.escape(s)
                                         for s in self.special_tokens) + ")"

        for text in iterable:
            if not text:
                continue
            # Split text by special tokens
            parts = re.split(special_pattern, text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    yield self.vocab_reverse[part.encode("utf-8")]
                else:
                    # Yield token IDs lazily from _encode_token
                    for token in pre_tokenize_text_iter(part):
                        yield from self._encode_token(token)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a string."""
        raise NotImplementedError("Decoding is not implemented yet")