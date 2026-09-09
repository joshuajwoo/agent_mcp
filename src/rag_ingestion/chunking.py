"""Sentence-aware chunking that retains source offsets for retrieval citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

SENTENCE_PATTERN = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.DOTALL)


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int
    end_char: int


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_sentence(sentence: TextChunk, max_words: int) -> list[TextChunk]:
    words = list(re.finditer(r"\S+", sentence.text))
    if len(words) <= max_words:
        return [sentence]
    parts: list[TextChunk] = []
    for start_index in range(0, len(words), max_words):
        end_index = min(start_index + max_words, len(words)) - 1
        start_char = sentence.start_char + words[start_index].start()
        end_char = sentence.start_char + words[end_index].end()
        parts.append(
            TextChunk(
                text=sentence.text[words[start_index].start() : words[end_index].end()],
                start_char=start_char,
                end_char=end_char,
            )
        )
    return parts


def chunk_text(text: str, *, max_words: int = 220, overlap_words: int = 40) -> list[TextChunk]:
    """Group sentences by length, carrying a small trailing-sentence overlap."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be non-negative and smaller than max_words")

    if not text.strip():
        return []
    units = [
        piece
        for sentence in SENTENCE_PATTERN.finditer(text)
        for piece in _split_long_sentence(
            TextChunk(
                text=sentence.group(), start_char=sentence.start(), end_char=sentence.end()
            ),
            max_words,
        )
        if piece
    ]

    chunk_ranges: list[tuple[int, int]] = []
    chunk_start = 0
    current_words = 0
    for index, unit in enumerate(units):
        unit_words = _word_count(unit.text)
        if current_words and current_words + unit_words > max_words:
            chunk_ranges.append((chunk_start, index))
            overlap_start = index
            overlap_count = 0
            while overlap_start > chunk_start:
                prior_words = _word_count(units[overlap_start - 1].text)
                if overlap_count + prior_words > overlap_words:
                    break
                overlap_start -= 1
                overlap_count += prior_words
            while overlap_start < index and overlap_count + unit_words > max_words:
                overlap_count -= _word_count(units[overlap_start].text)
                overlap_start += 1
            chunk_start = overlap_start
            current_words = overlap_count
        current_words += unit_words
    if units:
        chunk_ranges.append((chunk_start, len(units)))

    return [
        TextChunk(
            text=text[units[start].start_char : units[end - 1].end_char],
            start_char=units[start].start_char,
            end_char=units[end - 1].end_char,
        )
        for start, end in chunk_ranges
    ]
