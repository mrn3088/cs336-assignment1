from collections.abc import Iterator
import argparse
import os
from typing import BinaryIO
from dataclasses import dataclass, field
from collections import Counter
import regex as re
from concurrent.futures import ProcessPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)


def pre_tokenize_text_iter(text: str) -> Iterator[bytes]:
    """
    Pre-tokenize a text string into a sequence of bytes.
    """
    for m in PAT_RE.finditer(text):
        yield m.group(0).encode("utf-8")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@dataclass
class PreTokenizeChunkRequest:
    start: int
    end: int
    file_path: str
    special_tokens: list[str]


def pre_tokenize_chunk(request: PreTokenizeChunkRequest) -> Counter[bytes]:
    escaped = sorted((re.escape(s) for s in request.special_tokens), key=len, reverse=True)
    special_pattern = "(?:" + "|".join(escaped) + ")"

    with open(request.file_path, "rb") as f:
        f.seek(request.start)
        chunk = f.read(request.end - request.start).decode("utf-8", errors="ignore")
        parts = re.split(special_pattern, chunk)
        c = Counter()
        for part in parts:
            for b in pre_tokenize_text_iter(part):
                c[b] += 1
        return c


@dataclass
class PreTokenizationRequest:
    file_path: str
    num_processes: int = 4
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])


def run_pre_tokenization(request: PreTokenizationRequest) -> Counter[bytes]:
    """Run pre-tokenization on the file."""
    with open(request.file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, request.num_processes, request.special_tokens[0].encode("utf-8")
        )  # FIXME: Handle multiple special tokens
        print(f"Boundaries: {boundaries}")

        # Create chunk requests for parallel processing
        chunk_requests = [
            PreTokenizeChunkRequest(
                start=start, end=end, file_path=request.file_path, special_tokens=request.special_tokens
            )
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        # Process chunks in parallel (cap processes at num_processes)
        num_workers = min(len(chunk_requests), request.num_processes)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunk_counts = list(executor.map(pre_tokenize_chunk, chunk_requests))

        # Combine all counters
        total = Counter()
        for counts in chunk_counts:
            total += counts
        return total


def main():
    parser = argparse.ArgumentParser(description="Chunk a file for parallel pre-tokenization processing")
    parser.add_argument("file_path", type=str, help="Path to the file to process")
    parser.add_argument("--num-processes", type=int, default=4, help="Desired number of chunks/processes (default: 4)")
    parser.add_argument(
        "--split-token", type=str, default="<|endoftext|>", help="Special token to split on (default: '<|endoftext|>')"
    )

    args = parser.parse_args()

    request = PreTokenizationRequest(
        file_path=args.file_path, num_processes=args.num_processes, special_tokens=[args.split_token]
    )

    total_pre_token_counts = run_pre_tokenization(request)
    print(f"Total pre-tokens: {len(total_pre_token_counts)}")


if __name__ == "__main__":
    main()
