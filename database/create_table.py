from database.sqlite_db import connect


def create_tables() -> None:
    with connect() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id INTEGER PRIMARY KEY,
                company TEXT NOT NULL,
                year TEXT NOT NULL,
                revenue TEXT,
                net_income TEXT,
                operating_income TEXT,
                cash_flow TEXT,
                total_assets TEXT,
                total_liabilities TEXT,
                risk_factors TEXT,
                growth_drivers TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS report_chunks (
                source_file TEXT NOT NULL,
                position INTEGER NOT NULL,
                company TEXT NOT NULL,
                year TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (source_file, position)
            );

            CREATE INDEX IF NOT EXISTS report_chunks_model_company_year
            ON report_chunks (embedding_model, company COLLATE NOCASE, year);
        """)


if __name__ == "__main__":
    create_tables()
