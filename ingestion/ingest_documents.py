import argparse
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from config.settings import MARKDOWN_DIR, RAW_PDF_DIR
from database.create_table import create_tables
from database.save_metrics import save_metrics
from database.sqlite_db import connect
from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from llm.ollama_local import OllamaEmbeddings
from rag.kpi_extractor_rag import extract_financial_metrics, metrics_query
from vectorstore.sqlite_store import LocalVectorStore, rank_chunks


def parse_company_year(pdf_file: Path) -> tuple[str, int | None]:
    stem = pdf_file.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isascii() and parts[0].isdigit():
        return parts[-1] or stem, int(parts[0])
    if len(parts) >= 2 and parts[1].isascii() and parts[1].isdigit():
        return parts[0] or stem, int(parts[1])
    match = re.fullmatch(
        r"(.+?)[- ]+((?:19|20|21)\d{2})(?:[- ]+annual[- ]+report)?", stem, re.IGNORECASE
    )
    if match:
        return match[1], int(match[2])
    return stem, None


def ingest_document(
    pdf_path: str | Path,
    *,
    archive_to: Path | None = None,
    converted_markdown: Path | None = None,
) -> int:
    pdf_path = Path(pdf_path)
    company, year = parse_company_year(pdf_path)
    embeddings = OllamaEmbeddings()
    vector_store = LocalVectorStore(embeddings)
    create_tables()

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MARKDOWN_DIR) as markdown_dir:
        if converted_markdown is None:
            print(f"Converting {pdf_path.name} to Markdown", flush=True)
            markdown_file = Path(PDFToMarkdownConverter().convert_pdf(str(pdf_path), markdown_dir))
        else:
            markdown_file = Path(markdown_dir) / f"{pdf_path.stem}.md"
            shutil.copyfile(converted_markdown, markdown_file)
        chunks = chunk_markdown(markdown_file)

        if not chunks:
            raise ValueError("The PDF contains no extractable text.")

        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path.name

        print(f"Embedding {len(chunks)} passages from {pdf_path.name}", flush=True)
        vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
        query_vector = embeddings.embed_query(metrics_query(company, year))
        documents = rank_chunks(chunks, vectors, query_vector, top_k=10)
        print(f"Extracting metrics for {company} {year or ''}".rstrip(), flush=True)
        metrics = extract_financial_metrics(
            company, year, documents, report_markdown=markdown_file.read_text(encoding="utf-8")
        )

        with connect() as connection:
            vector_store.replace_report(connection, chunks, vectors, company, year, pdf_path.name)
            save_metrics(connection, company, year, metrics)
            if archive_to is not None:
                pdf_path.replace(archive_to)
            markdown_file.replace(MARKDOWN_DIR / markdown_file.name)

    print(f"Indexed {len(chunks)} chunks and saved metrics for {company} {year or ''}".rstrip())
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF or a directory of PDFs locally.")
    parser.add_argument("input_path", nargs="?", type=Path, default=RAW_PDF_DIR)
    args = parser.parse_args()
    input_path = args.input_path

    if input_path.is_dir():
        pdf_files = sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    elif input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdf_files = [input_path]
    else:
        parser.error(f"PDF or directory not found: {input_path}")

    if not pdf_files:
        parser.error(f"No PDFs found in {input_path}")
    for pdf_file in pdf_files:
        ingest_document(pdf_file)


if __name__ == "__main__":
    main()
