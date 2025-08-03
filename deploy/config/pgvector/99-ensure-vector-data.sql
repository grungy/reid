-- This script will always run and ensure that high_dim_vectors has data
-- It avoids complex PL/pgSQL to minimize potential errors

-- First make sure extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS high_dim_vectors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding VECTOR(384)
);

-- Simple approach to check if table is empty
DO $$
BEGIN
    -- If table is empty, insert simple data
    IF (SELECT COUNT(*) FROM high_dim_vectors) = 0 THEN
        -- First create a text representation of a 384-dimensional vector with all 0.1 values
        -- This is a simpler approach than using string manipulation
        WITH vector_text AS (
            SELECT '[' || string_agg('0.1', ',') || ']' AS vec_text
            FROM generate_series(1, 384)
        )
        -- Insert two sample vectors using the text representation
        INSERT INTO high_dim_vectors (name, embedding)
        SELECT 'Sample 384D Vector 1', vec_text::vector FROM vector_text
        UNION ALL
        SELECT 'Sample 384D Vector 2', vec_text::vector FROM vector_text;
    END IF;
END $$;
