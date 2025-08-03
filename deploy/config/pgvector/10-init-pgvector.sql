-- This file will be executed after pgvector is installed
-- Enable pgvector extension
DO $$
BEGIN
  -- Try to load the pgvector extension if available
  BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'pgvector extension successfully loaded!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension is not available: %', SQLERRM;
    RAISE NOTICE 'Continuing with setup but vector operations will not work.';
  END;
END
$$;

-- For testing purposes, create tables conditionally if vector extension exists
DO $$
BEGIN
  -- Check if the vector extension is installed
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    -- Create test tables for vectors
    CREATE TABLE IF NOT EXISTS test_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding VECTOR(3)  -- 3-dimensional vector for simplicity
    );

    -- Insert some sample data if table is empty
    IF (SELECT COUNT(*) FROM test_vectors) = 0 THEN
      INSERT INTO test_vectors (name, embedding)
      VALUES
          ('Sample Vector 1', '[0.1,0.2,0.3]'::vector),
          ('Sample Vector 2', '[0.5,0.4,0.3]'::vector);
      RAISE NOTICE 'Inserted sample data into test_vectors';
    END IF;

    -- Create another table with high-dimensional vectors
    CREATE TABLE IF NOT EXISTS high_dim_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding VECTOR(384)  -- 384-dimensional vector
    );

    RAISE NOTICE 'Vector tables created successfully';
  ELSE
    -- Create fallback tables without vector types for testing
    CREATE TABLE IF NOT EXISTS test_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding TEXT  -- Fallback to TEXT since vector is not available
    );

    -- Insert some sample data if table is empty
    IF (SELECT COUNT(*) FROM test_vectors) = 0 THEN
      INSERT INTO test_vectors (name, embedding)
      VALUES
          ('Sample Vector 1', '[0.1,0.2,0.3]'),
          ('Sample Vector 2', '[0.5,0.4,0.3]');
      RAISE NOTICE 'Inserted sample data into fallback test_vectors';
    END IF;

    -- Create another table with high-dimensional vectors as TEXT
    CREATE TABLE IF NOT EXISTS high_dim_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding TEXT  -- Fallback to TEXT since vector is not available
    );

    RAISE NOTICE 'Fallback text-based vector tables created';
  END IF;
END
$$;
