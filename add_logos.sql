
-- Run this in your Supabase SQL Editor to add the missing logo columns
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_logo VARCHAR(255);
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_logo VARCHAR(255);
