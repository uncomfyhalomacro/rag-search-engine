import re
from lib.semantic_search import CHUNK_SIZE, MAX_SEM_CHUNK_SIZE
from lib.hybrid_search import HYBRID_ALPHA


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def hybrid_score(bm25_score, semantic_score, alpha=HYBRID_ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalise_scores(scores):
    s0 = min(scores)
    s1 = max(scores)
    if s1 == s0:
        return [1.0 for _ in range(len(scores))]
    res = []
    for score in scores:
        r = (score - s0) / (s1 - s0)
        res.append(r)

    return res


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
    text = text.strip()
    if text == "":
        return []

    splits = re.split(r"(?<=[.!?])\s+", text)
    if len(splits) == 0:
        if text.strip() not in [".", "!", "?"]:
            return [text]

    chunks_storer = []
    start = 0
    for i in range(len(splits)):
        if start + size <= len(splits):
            if overlap > 0 and start + size + overlap <= len(splits):
                chunk = splits[start : start + size + overlap - 1]
                chunk[0] = chunk[0].lstrip()
                chunk[-1] = chunk[-1].rstrip()
                chunks_storer.append((start, chunk))
            else:
                chunk = splits[start : start + size]
                chunk[0] = chunk[0].lstrip()
                chunk[-1] = chunk[-1].rstrip()
                chunks_storer.append((start, chunk))
            start += size - overlap
        else:
            chunk = []
            if overlap > 0:
                chunk = splits[start:-overlap]
            else:
                chunk = splits[start:]
            if len(chunk) > 0:
                chunk[0] = chunk[0].lstrip()
                chunk[-1] = chunk[-1].rstrip()
                chunks_storer.append((start, chunk))
            break
    return chunks_storer
