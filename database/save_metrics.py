import sqlite3


def _as_lines(value: str | list[str] | None) -> str:
    if isinstance(value, list):
        return "\n".join(value)
    return value or ""


def save_metrics(
    connection: sqlite3.Connection,
    company: str,
    year: int | None,
    metrics: dict,
) -> None:
    query = """
        INSERT INTO financial_metrics (
            company, year, revenue, net_income, operating_income, cash_flow,
            total_assets, total_liabilities, risk_factors, growth_drivers
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    connection.execute(query, (
        company,
        str(year) if year is not None else "",
        metrics.get("revenue"),
        metrics.get("net_income"),
        metrics.get("operating_income"),
        metrics.get("cash_flow"),
        metrics.get("total_assets"),
        metrics.get("total_liabilities"),
        _as_lines(metrics.get("risk_factors")),
        _as_lines(metrics.get("growth_drivers")),
    ))
