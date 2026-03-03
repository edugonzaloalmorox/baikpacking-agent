CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,    
    url TEXT,               
    body TEXT,              
    raw JSONB               
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url ON articles (url);

CREATE TABLE IF NOT EXISTS riders (
    id SERIAL PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,

    -- rider fields
    name TEXT,
    age INT,
    location TEXT,
    bike TEXT,
    key_items TEXT,
    frame_type TEXT,
    frame_material TEXT,
    wheel_size TEXT,
    tyre_width TEXT,
    electronic_shifting BOOLEAN,

    raw JSONB            
);


CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.rider_chunks (
  id           BIGSERIAL PRIMARY KEY,
  rider_id     BIGINT NOT NULL REFERENCES public.riders(id) ON DELETE CASCADE,

  -- chunk identity
  chunk_kind   TEXT NOT NULL,         
  chunk_ix     INT NOT NULL,         
  chunk_text   TEXT NOT NULL,
  chunk_tokens INT,

  -- embedding
  embedding    vector(1024),
  model        TEXT NOT NULL DEFAULT 'text-embedding-3-small',

  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (rider_id, chunk_kind, chunk_ix)
);

CREATE INDEX IF NOT EXISTS rider_chunks_rider_id_idx
  ON public.rider_chunks (rider_id);

CREATE INDEX IF NOT EXISTS rider_chunks_embedding_ivfflat
  ON public.rider_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);

