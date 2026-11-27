import argparse

from database.create_table import create_tables
from llm.ollama_local import OllamaEmbeddings
from vectorstore.sqlite_store import LocalVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect passages retrieved from the local index.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--company")
    parser.add_argument("--year", type=int)
    args = parser.parse_args()

    create_tables()
    store = LocalVectorStore(OllamaEmbeddings())
    documents = store.invoke(" ".join(args.query), company=args.company, year=args.year)
    print(f"Found {len(documents)} passages")
    for index, document in enumerate(documents, start=1):
        print(f"\n{index}. {document.metadata['source_file']}")
        print(" ".join(document.page_content.split())[:350])


if __name__ == "__main__":
    main()
