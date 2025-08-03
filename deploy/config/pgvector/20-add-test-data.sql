-- This file adds some test high-dimensional vector data

-- Try to add vector data if pgvector is installed, otherwise add text data
DO $$
DECLARE
    dimension_count INTEGER := 384;
    repeated_values_needed INTEGER;
    vector_prefix TEXT;
    vector_middle TEXT;
    vector_suffix TEXT;
    full_vector TEXT;
    vector_prefix2 TEXT;
    vector_middle2 TEXT;
    vector_suffix2 TEXT;
    full_vector2 TEXT;
    vector_prefix3 TEXT;
    vector_middle3 TEXT;
    vector_suffix3 TEXT;
    full_vector3 TEXT;
    vector_prefix4 TEXT;
    vector_middle4 TEXT;
    vector_suffix4 TEXT;
    full_vector4 TEXT;
    vector_prefix5 TEXT;
    vector_middle5 TEXT;
    vector_suffix5 TEXT;
    full_vector5 TEXT;
    vector_prefix6 TEXT;
    vector_middle6 TEXT;
    vector_suffix6 TEXT;
    full_vector6 TEXT;
    vector_prefix7 TEXT;
    vector_middle7 TEXT;
    vector_suffix7 TEXT;
    full_vector7 TEXT;
    vector_prefix8 TEXT;
    vector_middle8 TEXT;
    vector_suffix8 TEXT;
    full_vector8 TEXT;
    vector_prefix9 TEXT;
    vector_middle9 TEXT;
    vector_suffix9 TEXT;
    full_vector9 TEXT;
    vector_prefix10 TEXT;
    vector_middle10 TEXT;
    vector_suffix10 TEXT;
    full_vector10 TEXT;
BEGIN
  -- Check if the vector extension is available
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    -- Only proceed if the high_dim_vectors table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'high_dim_vectors') THEN

      -- Calculate how many times to repeat the 10-value pattern (taking into account prefix and suffix)
      repeated_values_needed := (dimension_count - 14) / 10;

      -- First embedding - Small values
      vector_prefix := '[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,';
      vector_middle := repeat('0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,', repeated_values_needed);
      vector_suffix := '0.01,0.02,0.03,0.95]'; -- Last value much larger than first
      full_vector := vector_prefix || vector_middle || vector_suffix;

      -- Second embedding - Medium values
      vector_prefix2 := '[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.10,';
      vector_middle2 := repeat('0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.10,', repeated_values_needed);
      vector_suffix2 := '0.10,0.20,0.30,0.80]'; -- Last value larger than first
      full_vector2 := vector_prefix2 || vector_middle2 || vector_suffix2;

      -- Third embedding - Increasing pattern first to last
      vector_prefix3 := '[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,';
      vector_middle3 := repeat('0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,', repeated_values_needed);
      vector_suffix3 := '1.05,1.10,1.15,1.50]'; -- Last value much larger than first
      full_vector3 := vector_prefix3 || vector_middle3 || vector_suffix3;

      -- Fourth embedding - Decreasing pattern first to last
      vector_prefix4 := '[0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,';
      vector_middle4 := repeat('0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05,0.01,', repeated_values_needed);
      vector_suffix4 := '0.01,0.01,0.01,0.01]'; -- Last value smaller than first
      full_vector4 := vector_prefix4 || vector_middle4 || vector_suffix4;

      -- Fifth embedding - First value high, last value low (big difference)
      vector_prefix5 := '[0.99,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,';
      vector_middle5 := repeat('0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,', repeated_values_needed);
      vector_suffix5 := '0.01,0.01,0.01,0.01]'; -- Last value equal to others but much smaller than first
      full_vector5 := vector_prefix5 || vector_middle5 || vector_suffix5;

      -- Sixth embedding - First value low, last value high (big difference)
      vector_prefix6 := '[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,';
      vector_middle6 := repeat('0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,', repeated_values_needed);
      vector_suffix6 := '0.01,0.01,0.01,0.99]'; -- Last value much higher
      full_vector6 := vector_prefix6 || vector_middle6 || vector_suffix6;

      -- Seventh embedding - Sine wave pattern with high at end
      vector_prefix7 := '[0.50,0.60,0.70,0.80,0.90,1.00,0.90,0.80,0.70,0.60,';
      vector_middle7 := repeat('0.50,0.40,0.30,0.20,0.10,0.00,0.10,0.20,0.30,0.40,', repeated_values_needed);
      vector_suffix7 := '0.50,0.60,0.70,0.95]'; -- Last value higher than first
      full_vector7 := vector_prefix7 || vector_middle7 || vector_suffix7;

      -- Eighth embedding - Sine wave pattern with low at end
      vector_prefix8 := '[0.50,0.40,0.30,0.20,0.10,0.00,0.10,0.20,0.30,0.40,';
      vector_middle8 := repeat('0.50,0.60,0.70,0.80,0.90,1.00,0.90,0.80,0.70,0.60,', repeated_values_needed);
      vector_suffix8 := '0.50,0.40,0.30,0.05]'; -- Last value lower than first
      full_vector8 := vector_prefix8 || vector_middle8 || vector_suffix8;

      -- Ninth embedding - First and last identical, middle varies
      vector_prefix9 := '[0.42,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,';
      vector_middle9 := repeat('0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.10,', repeated_values_needed);
      vector_suffix9 := '0.10,0.20,0.30,0.42]'; -- Same as first value
      full_vector9 := vector_prefix9 || vector_middle9 || vector_suffix9;

      -- Tenth embedding - Large values
      vector_prefix10 := '[1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.20,2.30,2.40,';
      vector_middle10 := repeat('1.50,1.60,1.70,1.80,1.90,2.00,2.10,2.20,2.30,2.40,', repeated_values_needed);
      vector_suffix10 := '1.50,1.60,1.70,3.00]'; -- Last value much higher
      full_vector10 := vector_prefix10 || vector_middle10 || vector_suffix10;

      BEGIN
        -- Insert vectors that don't already exist by using the name as a key

        -- Insert 'Small values, larger last' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Small values, larger last') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Small values, larger last', full_vector::vector);
          RAISE NOTICE 'Added "Small values, larger last" vector';
        END IF;

        -- Insert 'Medium values, larger last' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Medium values, larger last') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Medium values, larger last', full_vector2::vector);
          RAISE NOTICE 'Added "Medium values, larger last" vector';
        END IF;

        -- Insert 'Increasing pattern' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Increasing pattern') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Increasing pattern', full_vector3::vector);
          RAISE NOTICE 'Added "Increasing pattern" vector';
        END IF;

        -- Insert 'Decreasing pattern' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Decreasing pattern') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Decreasing pattern', full_vector4::vector);
          RAISE NOTICE 'Added "Decreasing pattern" vector';
        END IF;

        -- Insert 'High first, low last' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'High first, low last') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('High first, low last', full_vector5::vector);
          RAISE NOTICE 'Added "High first, low last" vector';
        END IF;

        -- Insert 'Low first, high last' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Low first, high last') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Low first, high last', full_vector6::vector);
          RAISE NOTICE 'Added "Low first, high last" vector';
        END IF;

        -- Insert 'Sine wave with high end' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Sine wave with high end') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Sine wave with high end', full_vector7::vector);
          RAISE NOTICE 'Added "Sine wave with high end" vector';
        END IF;

        -- Insert 'Sine wave with low end' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Sine wave with low end') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Sine wave with low end', full_vector8::vector);
          RAISE NOTICE 'Added "Sine wave with low end" vector';
        END IF;

        -- Insert 'Same first and last' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Same first and last') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Same first and last', full_vector9::vector);
          RAISE NOTICE 'Added "Same first and last" vector';
        END IF;

        -- Insert 'Large values, much larger last' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Large values, much larger last') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Large values, much larger last', full_vector10::vector);
          RAISE NOTICE 'Added "Large values, much larger last" vector';
        END IF;

      EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Error adding high-dimensional vector data: %', SQLERRM;
      END;
    END IF;
  ELSE
    -- Vector extension not available, insert as TEXT
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'high_dim_vectors') THEN

      BEGIN
        -- Insert vectors as text strings if they don't already exist

        -- Insert 'Small values, larger last (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Small values, larger last (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Small values, larger last (TEXT)', '[0.01,0.02,0.03,0.04,0.05,...,0.95]');
          RAISE NOTICE 'Added "Small values, larger last (TEXT)" vector';
        END IF;

        -- Insert 'Medium values, larger last (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Medium values, larger last (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Medium values, larger last (TEXT)', '[0.10,0.20,0.30,0.40,0.50,...,0.80]');
          RAISE NOTICE 'Added "Medium values, larger last (TEXT)" vector';
        END IF;

        -- Insert 'Increasing pattern (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Increasing pattern (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Increasing pattern (TEXT)', '[0.05,0.10,0.15,0.20,0.25,...,1.50]');
          RAISE NOTICE 'Added "Increasing pattern (TEXT)" vector';
        END IF;

        -- Insert 'Decreasing pattern (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Decreasing pattern (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Decreasing pattern (TEXT)', '[0.95,0.90,0.85,0.80,0.75,...,0.01]');
          RAISE NOTICE 'Added "Decreasing pattern (TEXT)" vector';
        END IF;

        -- Insert 'High first, low last (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'High first, low last (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('High first, low last (TEXT)', '[0.99,0.01,0.01,0.01,0.01,...,0.01]');
          RAISE NOTICE 'Added "High first, low last (TEXT)" vector';
        END IF;

        -- Insert 'Low first, high last (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Low first, high last (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Low first, high last (TEXT)', '[0.01,0.01,0.01,0.01,0.01,...,0.99]');
          RAISE NOTICE 'Added "Low first, high last (TEXT)" vector';
        END IF;

        -- Insert 'Sine wave with high end (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Sine wave with high end (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Sine wave with high end (TEXT)', '[0.50,0.60,0.70,0.80,0.90,...,0.95]');
          RAISE NOTICE 'Added "Sine wave with high end (TEXT)" vector';
        END IF;

        -- Insert 'Sine wave with low end (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Sine wave with low end (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Sine wave with low end (TEXT)', '[0.50,0.40,0.30,0.20,0.10,...,0.05]');
          RAISE NOTICE 'Added "Sine wave with low end (TEXT)" vector';
        END IF;

        -- Insert 'Same first and last (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Same first and last (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Same first and last (TEXT)', '[0.42,0.10,0.20,0.30,0.40,...,0.42]');
          RAISE NOTICE 'Added "Same first and last (TEXT)" vector';
        END IF;

        -- Insert 'Large values, much larger last (TEXT)' if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM high_dim_vectors WHERE name = 'Large values, much larger last (TEXT)') THEN
          INSERT INTO high_dim_vectors (name, embedding)
          VALUES ('Large values, much larger last (TEXT)', '[1.50,1.60,1.70,1.80,1.90,...,3.00]');
          RAISE NOTICE 'Added "Large values, much larger last (TEXT)" vector';
        END IF;

        RAISE NOTICE 'Added test high-dimensional vector data as TEXT';
      EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Error adding high-dimensional text data: %', SQLERRM;
      END;
    END IF;
  END IF;
END $$;
