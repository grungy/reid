-- Check if pgvector extension is already available
DO $$
BEGIN
  -- Try to load the pgvector extension if available
  BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'pgvector extension successfully loaded!';
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension is not available. You may need to compile and install it.';
    RAISE NOTICE 'Error: %', SQLERRM;
  END;
END
$$;

-- Create a test table for vectors if the extension is available
DO $$
BEGIN
  -- Check if the vector extension is installed
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    -- Create test table
    CREATE TABLE IF NOT EXISTS test_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding VECTOR(3)  -- 3-dimensional vector for simplicity
    );

    -- Add sample data if table is empty
    IF (SELECT COUNT(*) FROM test_vectors) = 0 THEN
      INSERT INTO test_vectors (name, embedding)
      VALUES
          ('Sample Vector 1', '[0.1,0.2,0.3]'::vector),
          ('Sample Vector 2', '[0.5,0.4,0.3]'::vector);
      RAISE NOTICE 'Sample vector data added';
    END IF;

    -- Create high-dimensional table for actual use
    CREATE TABLE IF NOT EXISTS high_dim_vectors (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        embedding VECTOR(384)  -- 384-dimensional vector
    );

    -- Always ensure high_dim_vectors has data by inserting if empty
    IF (SELECT COUNT(*) FROM high_dim_vectors) = 0 THEN
      -- Use a simpler approach directly in this script
      DECLARE
        dim_count INTEGER := 384;
        test_vector TEXT := '[' ||
                         repeat('0.1,', dim_count-1) ||
                         '0.1]';
      BEGIN
        -- Insert sample data
        INSERT INTO high_dim_vectors (name, embedding)
        VALUES
          ('Sample 384D Vector', test_vector::vector),
          ('Sample 384D Vector 2', test_vector::vector);
        RAISE NOTICE 'Added high-dimensional vector data';
      EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Error adding high-dimensional vector data: %', SQLERRM;
      END;
    END IF;

    RAISE NOTICE 'Vector tables created successfully';
  ELSE
    RAISE NOTICE 'Vector extension not available - skipping vector table creation';
  END IF;
END
$$;
