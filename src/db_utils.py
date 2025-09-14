import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError("Missing DB_URI in .env file")

def get_connection():
    return psycopg2.connect(DB_URI)

def fetch_one(query, params=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params or ())
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

def fetch_all(query, params=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params or ())
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def execute(query, params=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params or ())
    conn.commit()
    cur.close()
    conn.close()
