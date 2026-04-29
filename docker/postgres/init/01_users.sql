-- Grants for analyst_bot (read-only user for READ_DATABASE_URL).
-- analyst_bot is created in 00_roles.sql — only grants are set here.

GRANT CONNECT ON DATABASE ecommerce TO analyst_bot;
GRANT USAGE ON SCHEMA public TO analyst_bot;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_bot;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analyst_bot;
