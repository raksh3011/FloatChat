import os
import psycopg2
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError("Missing DB_URI in .env file")

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(DB_URI)
cur = conn.cursor()

cur.execute("""
    SELECT profile_id, short_summary || ' ' || (
        SELECT string_agg(v.variable_name || ': ' || m.value::text, '; ')
        FROM measurements m
        JOIN measurement_variables v ON m.variable_id = v.variable_id
        WHERE m.profile_id = p.profile_id
        LIMIT 10
    ) AS text
    FROM profiles p;
""")
profiles = cur.fetchall()

for pid, text in profiles:
    if text:
        embedding = model.encode(text, normalize_embeddings=True)
        cur.execute(
            "UPDATE profiles SET embedding = %s WHERE profile_id = %s",
            (embedding.tolist(), pid),
        )

conn.commit()
cur.close()
conn.close()
print("✅ Embeddings stored.")
