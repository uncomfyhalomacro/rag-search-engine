import re
from lib.semantic_search import CHUNK_SIZE, MAX_SEM_CHUNK_SIZE


def ordinary_chunker(text, size=CHUNK_SIZE, overlap=0):
    chunks_storer = []
    initial_chunks = text.split(None, maxsplit=size)
    while len(initial_chunks) > 0:
        if len(initial_chunks) == 1:
            chunks_storer[-1] += " " + initial_chunks.pop()
        else:
            chunk = " ".join(initial_chunks[:-1])
            chunks_storer.append(chunk)
            initial_chunks = initial_chunks[-1].split(None, maxsplit=size)
            if overlap > 0:
                if len(initial_chunks) > overlap:
                    overlap = chunk.rsplit(None, maxsplit=overlap)[1:]
                    overlap.extend(initial_chunks)
                    initial_chunks = overlap

    return chunks_storer


def semantic_chunker(text, size=MAX_SEM_CHUNK_SIZE, overlap=0):
    splits = re.split(r"(?<=[.!?])\s+", text)
    chunks_storer = []
    start = 0
    for i in range(len(splits)):
        if start + size <= len(splits):
            if overlap > 0 and start + size + overlap <= len(splits):
                chunk = splits[start : start + size + overlap - 1]
                chunks_storer.append((start, chunk))
            else:
                chunk = splits[start : start + size]
                chunks_storer.append((start, chunk))
            start += size - overlap
        else:
            chunk = []
            if overlap > 0:
                chunk = splits[start:-overlap]
            else:
                chunk = splits[start:]
            if len(chunk) > 0:
                chunks_storer.append((start, chunk))
            break
    return chunks_storer
