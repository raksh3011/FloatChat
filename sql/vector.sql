-- Applied only when the pgvector extension can be created.
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS embedding vector(384);
-- Approximate nearest-neighbour search over cosine distance (all-MiniLM-L6-v2 embeddings are normalised).
CREATE INDEX IF NOT EXISTS idx_profiles_embedding_hnsw
    ON profiles USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
