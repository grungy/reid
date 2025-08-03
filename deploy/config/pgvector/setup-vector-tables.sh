#!/bin/bash
set -e

echo "Setting up vector tables..."

# For debugging - show current schemas and tables
echo "Current tables before setup:"
psql -U postgres -c "\dt"

# First ensure the extension exists
echo "Ensuring vector extension is installed..."
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Check if the extension is loaded
echo "Verifying vector extension status:"
psql -U postgres -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

# Create the high_dim_vectors table if it doesn't exist - explicitly use public schema
echo "Creating high_dim_vectors table if it doesn't exist..."
psql -U postgres << EOF
CREATE TABLE IF NOT EXISTS public.high_dim_vectors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding VECTOR(384)
);
EOF

# Check if the table exists
echo "Checking if table was created properly:"
psql -U postgres -c "\dt public.high_dim_vectors"

# Check if the table is empty and populate it if needed
echo "Populating table with vector data if empty..."
psql -U postgres << EOF
DO \$\$
BEGIN
    -- If table is empty, insert simple data
    IF (SELECT COUNT(*) FROM public.high_dim_vectors) = 0 THEN
        -- First create a text representation of a 384-dimensional vector with all 0.1 values
        WITH vector_text AS (
            SELECT '[' || string_agg('0.1', ',') || ']' AS vec_text
            FROM generate_series(1, 384)
        )
        -- Insert sample vectors using the text representation
        INSERT INTO public.high_dim_vectors (name, embedding)
        SELECT 'Sample 384D Vector 1', vec_text::vector FROM vector_text
        UNION ALL
        SELECT 'Sample 384D Vector 2', vec_text::vector FROM vector_text;

        RAISE NOTICE 'Populated vector table with initial data';
    ELSE
        RAISE NOTICE 'Vector table already has data - keeping existing records';
    END IF;
END \$\$;
EOF

# Finally verify what was created
echo "Current vector table contents:"
psql -U postgres -c "SELECT count(*) FROM public.high_dim_vectors;"
psql -U postgres -c "SELECT id, name, substr(embedding::text, 1, 30) || '...' FROM public.high_dim_vectors LIMIT 2;"

# Show all tables for verification
echo "All tables after setup:"
psql -U postgres -c "\dt"

echo "Vector tables setup complete!"
