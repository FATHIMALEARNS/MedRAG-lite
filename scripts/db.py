import sqlite3

def get_connection():
    return sqlite3.connect("medrag.db", check_same_thread=False)

def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()
