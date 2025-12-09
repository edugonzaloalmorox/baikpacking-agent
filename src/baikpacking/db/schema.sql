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
