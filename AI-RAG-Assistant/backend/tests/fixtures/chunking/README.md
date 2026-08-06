# Chunking strategy fixtures

Deterministic files for manual and automated validation of RAG chunking.

## Sentence_Chunker_Test.txt

- **Size:** ~4.1 KB (12 numbered sentences, 300–350 chars each)
- **Strategy:** `sentence`
- **Expected `indexed_chunk_count`:** `3` (with default `CHUNK_SIZE=1500`)
- **Invariant:** no original sentence is split across chunks

## Markdown_Header_Chunker_Test.txt

Plain text file using markdown-style `#` headings (works with Header Aware chunking on `.txt` uploads).

- **Size:** ~7.0 KB
- **Strategy:** `header_aware`
- **Expected `indexed_chunk_count`:** `6`

| Section        | Expected chunks | Notes                          |
|----------------|-----------------|--------------------------------|
| Introduction   | 1               | entire section ≤ CHUNK_SIZE    |
| Embeddings     | 2               | `# Embeddings` repeated        |
| Retrieval      | 2               | `# Retrieval` repeated         |
| Conclusion     | 1               | entire section ≤ CHUNK_SIZE    |

## Quick check

```bash
cd backend
.venv/bin/pytest tests/chunking/test_chunking_fixtures.py -v
```

Upload with **Advanced RAG options** enabled and confirm `indexed_chunk_count` in the upload response.
