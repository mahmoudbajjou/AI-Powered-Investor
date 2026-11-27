import sqlite3
from contextlib import closing, contextmanager

from config.settings import DATABASE_PATH


@contextmanager
def connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH, timeout=15)) as connection, connection:
        connection.row_factory = sqlite3.Row
        yield connection
