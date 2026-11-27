from database.sqlite_db import connect


def get_metrics() -> list[dict]:
    query = """
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY company, year
                       ORDER BY created_at DESC, id DESC
                   ) AS rn
            FROM financial_metrics
        ) AS metrics
        WHERE rn = 1
        ORDER BY company, year DESC
    """
    with connect() as connection:
        return [dict(row) for row in connection.execute(query)]
