import sqlite3
from collections.abc import Sequence

import numpy as np
from langchain_core.documents import Document

from database.sqlite_db import connect
from llm.ollama_local import OllamaEmbeddings


def rank_chunks(
    chunks: Sequence[Document],
    vectors: Sequence,
    query_vector: Sequence[float],
    top_k: int,
) -> list[Document]:
    if not chunks:
        return []

    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if matrix.shape != (len(chunks), query.size):
        raise ValueError("Embedding dimensions differ. Reindex the reports with the current model.")

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.full(len(chunks), -np.inf)
    np.divide(matrix @ query, norms, out=scores, where=norms > 0)
    best = np.argsort(-scores, kind="stable")[:top_k]
    return [chunks[i] for i in best if np.isfinite(scores[i])]


class LocalVectorStore:
    def __init__(self, embeddings: OllamaEmbeddings):
        self.embeddings = embeddings

    def replace_report(
        self,
        connection: sqlite3.Connection,
        chunks: Sequence[Document],
        vectors: Sequence[Sequence[float]],
        company: str,
        year: int | None,
        source_file: str,
    ) -> None:
        connection.execute(
            "DELETE FROM report_chunks WHERE source_file = ?", (source_file,)
        )
        year_text = str(year) if year is not None else ""
        rows = (
            (
                source_file, position, company, year_text, chunk.page_content,
                self.embeddings.model, np.asarray(vector, dtype="<f4").tobytes(),
            )
            for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        )
        connection.executemany("""
            INSERT INTO report_chunks (
                source_file, position, company, year, content, embedding_model, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | None = None,
        top_k: int = 6,
    ) -> list[Document]:
        sql = """
            SELECT source_file, company, year, content, embedding
            FROM report_chunks
            WHERE embedding_model = ?
        """
        params = [self.embeddings.model]
        if company:
            sql += " AND company = ? COLLATE NOCASE"
            params.append(company)
        if year is not None:
            sql += " AND year = ?"
            params.append(str(year))
        sql += " ORDER BY source_file, position"

        with connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        if not rows:
            return []

        chunks = [
            Document(
                page_content=row["content"],
                metadata={key: row[key] for key in ("source_file", "company", "year")},
            )
            for row in rows
        ]
        vectors = [np.frombuffer(row["embedding"], dtype="<f4") for row in rows]
        return rank_chunks(chunks, vectors, self.embeddings.embed_query(query), top_k)
