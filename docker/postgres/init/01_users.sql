-- Create the read-only analyst user used by READ_DATABASE_URL.
-- This script runs automatically on first container creation.

CREATE USER analyst_bot WITH PASSWORD 'root123';
GRANT CONNECT ON DATABASE ecommerce TO analyst_bot;
GRANT USAGE ON SCHEMA public TO analyst_bot;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_bot;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analyst_bot;
