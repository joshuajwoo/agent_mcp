from rag_ingestion.chunking import chunk_text


def test_chunking_keeps_sentence_boundaries_and_overlap() -> None:
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."

    chunks = chunk_text(text, max_words=7, overlap_words=3)

    assert [chunk.text for chunk in chunks] == [
        "One two three. Four five six.",
        "Four five six. Seven eight nine.",
        "Seven eight nine. Ten eleven twelve.",
    ]
    assert all(text[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks)


def test_chunking_splits_a_sentence_longer_than_limit() -> None:
    chunks = chunk_text("one two three four five", max_words=2, overlap_words=0)

    assert [chunk.text for chunk in chunks] == ["one two", "three four", "five"]


def test_chunk_offsets_reference_original_text() -> None:
    text = "  First sentence.\n\nSecond sentence.  "

    chunks = chunk_text(text, max_words=10, overlap_words=0)

    assert text[chunks[0].start_char : chunks[0].end_char] == chunks[0].text
