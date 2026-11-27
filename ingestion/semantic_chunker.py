import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config.settings import MARKDOWN_DIR
from llm.ollama_local import OllamaEmbeddings


def read_markdown(markdown_file: str | Path) -> str:
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str | Path,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    text = read_markdown(markdown_file)
    if not text.strip():
        return []

    if embeddings is None:
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=2200, chunk_overlap=200
        )
        return splitter.create_documents([text])

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        sentence_split_regex=r"(?<=[.!?])\s+|\n[ \t]*\n",
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2200,
        chunk_overlap=200,
    )
    semantic_chunks = semantic_splitter.create_documents([text])
    return size_splitter.split_documents(semantic_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a Markdown report using local embeddings.")
    parser.add_argument("markdown_file", nargs="?", type=Path, default=MARKDOWN_DIR / "2024_Apple.md")
    args = parser.parse_args()

    chunks = chunk_markdown(args.markdown_file, OllamaEmbeddings())
    print(f"Generated {len(chunks)} chunks")
    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index} ({len(chunk.page_content)} characters)")
        print(chunk.page_content[:1000])


if __name__ == "__main__":
    main()
