from collections.abc import Iterable

from langchain_core.documents import Document


def build_context(documents: Iterable[Document], max_chars: int = 12_000) -> str:
    parts = []
    remaining = max_chars
    for doc in documents:
        source = doc.metadata.get("source_file", "Financial report")
        header = f"Source: {source}\n"
        if remaining <= len(header):
            break
        part = header + doc.page_content[:remaining - len(header)]
        parts.append(part)
        remaining -= len(part) + 2
    return "\n\n".join(parts)
