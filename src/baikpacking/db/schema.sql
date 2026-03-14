CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- articles
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.articles (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    url         TEXT NOT NULL,
    body        TEXT,
    raw         JSONB,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT articles_url_unique UNIQUE (url)
);

CREATE INDEX IF NOT EXISTS idx_articles_title
    ON public.articles (title);

-- ============================================================================
-- riders
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.riders (
    id                   BIGSERIAL PRIMARY KEY,
    article_id           BIGINT NOT NULL REFERENCES public.articles(id) ON DELETE CASCADE,

    name                 TEXT,
    age                  INT,
    location             TEXT,
    bike                 TEXT,
    key_items            TEXT,
    frame_type           TEXT,
    frame_material       TEXT,
    wheel_size           TEXT,
    tyre_width           TEXT,
    electronic_shifting  BOOLEAN,

    raw                  JSONB,

    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT riders_age_nonnegative CHECK (age IS NULL OR age >= 0)
);

CREATE INDEX IF NOT EXISTS idx_riders_article_id
    ON public.riders (article_id);

CREATE INDEX IF NOT EXISTS idx_riders_name
    ON public.riders (name);

CREATE INDEX IF NOT EXISTS idx_riders_location
    ON public.riders (location);

-- Optional: to prevent accidental duplicates within the same article for named riders only.
 
 CREATE UNIQUE INDEX IF NOT EXISTS idx_riders_article_name_unique
     ON public.riders (article_id, lower(name))
     WHERE name IS NOT NULL;

-- ============================================================================
-- rider_chunks
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.rider_chunks (
    id           BIGSERIAL PRIMARY KEY,
    rider_id      BIGINT NOT NULL REFERENCES public.riders(id) ON DELETE CASCADE,

    chunk_kind    TEXT NOT NULL,
    chunk_ix      INT NOT NULL,
    chunk_text    TEXT NOT NULL,
    chunk_tokens  INT,

    embedding     vector(1024),
    model         TEXT NOT NULL DEFAULT 'ollama',

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT rider_chunks_unique UNIQUE (rider_id, chunk_kind, chunk_ix),
    CONSTRAINT rider_chunks_chunk_ix_nonnegative CHECK (chunk_ix >= 0),
    CONSTRAINT rider_chunks_chunk_tokens_nonnegative CHECK (chunk_tokens IS NULL OR chunk_tokens >= 0)
);

CREATE INDEX IF NOT EXISTS idx_rider_chunks_rider_id
    ON public.rider_chunks (rider_id);

CREATE INDEX IF NOT EXISTS idx_rider_chunks_chunk_kind
    ON public.rider_chunks (chunk_kind);

CREATE INDEX IF NOT EXISTS idx_rider_chunks_model
    ON public.rider_chunks (model);

CREATE INDEX IF NOT EXISTS idx_rider_chunks_embedding_ivfflat
    ON public.rider_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 200);