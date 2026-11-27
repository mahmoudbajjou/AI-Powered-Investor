import argparse
import sqlite3
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config.settings import MARKDOWN_DIR
from llm.ollama_local import OllamaEmbeddings, get_structured_completion
from rag.context import build_context
from rag.financial_tables import AMOUNT_FIELDS, extract_table_amounts
from vectorstore.sqlite_store import LocalVectorStore


class ReportNarrative(BaseModel):
    risk_factors: str | list[str] | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list[str] | None = Field(None, alias="Top Growth Drivers")


class FinancialMetrics(ReportNarrative):
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(None, alias="Cash Flow from Operating Activities")
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")


def metrics_query(company: str, year: int | None) -> str:
    return (
        f"{company} {year or ''} annual report: principal risk factors, "
        "business outlook, and reasons for revenue growth."
    )


def extract_financial_metrics(
    company: str, year: int | None, documents: list[Document], *, report_markdown: str
) -> dict:
    metrics = FinancialMetrics().model_dump()
    metrics.update(extract_table_amounts(report_markdown, year))
    context = build_context(documents)
    if not context:
        return metrics

    prompt = f"""
Summarize the principal risk factors and growth drivers for {company},
fiscal year {year or 'shown in the report'}.

Use only the report extracts below. Be concise and use null if the extracts
do not support an answer. Return the fields specified by the JSON schema.

Report extracts:
{context}
"""
    narrative = get_structured_completion(prompt, ReportNarrative)
    return {**metrics, **narrative.model_dump()}


def _refresh_amounts(
    connection: sqlite3.Connection, company: str, year: int, amounts: dict[str, str]
) -> dict:
    from database.save_metrics import save_metrics

    previous = connection.execute(
        """
        SELECT * FROM financial_metrics
        WHERE company = ? AND year = ?
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (company, str(year)),
    ).fetchone()
    metrics = FinancialMetrics().model_dump()
    metrics.update(amounts)
    if previous is not None:
        metrics.update({field: previous[field] for field in ReportNarrative.model_fields})
    if previous is None or any(previous[field] != metrics[field] for field in AMOUNT_FIELDS):
        save_metrics(connection, company, year, metrics)
    return metrics


def main() -> None:
    from database.create_table import create_tables
    from database.sqlite_db import connect

    parser = argparse.ArgumentParser(description="Extract metrics from an indexed report locally.")
    parser.add_argument("company", nargs="?", default="Apple")
    parser.add_argument("year", nargs="?", type=int, default=2024)
    parser.add_argument(
        "--refresh-amounts", action="store_true", help="Save verified amounts from the indexed report without Ollama."
    )
    args = parser.parse_args()

    create_tables()
    with connect() as connection:
        if args.refresh_amounts:
            connection.execute("BEGIN IMMEDIATE")
        reports = connection.execute(
            "SELECT DISTINCT company, source_file FROM report_chunks WHERE company = ? COLLATE NOCASE AND year = ?",
            (args.company, str(args.year)),
        ).fetchall()
        if not reports:
            parser.error("No report found for this company and year. Index the PDF first.")
        if len(reports) > 1:
            parser.error("More than one report matches this company and year.")
        company, source_file = reports[0]
        markdown_file = MARKDOWN_DIR / Path(source_file).with_suffix(".md").name
        if not markdown_file.is_file():
            parser.error(f"Report Markdown is missing: {markdown_file}")
        markdown = markdown_file.read_text(encoding="utf-8")

        if args.refresh_amounts:
            amounts = extract_table_amounts(markdown, args.year)
            if not amounts:
                parser.error("No monetary amounts could be verified in the report tables.")
            metrics = _refresh_amounts(connection, company, args.year, amounts)

    if not args.refresh_amounts:
        store = LocalVectorStore(OllamaEmbeddings())
        documents = store.invoke(
            metrics_query(company, args.year),
            company=company,
            year=args.year,
            top_k=10,
        )
        metrics = extract_financial_metrics(company, args.year, documents, report_markdown=markdown)
    print(FinancialMetrics.model_validate(metrics, by_name=True).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
