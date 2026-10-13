import sqlite3
import os

DATABASE = "database/lungvision.db"

def get_db_connection():
    print("Using database:", os.path.abspath(DATABASE))
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn