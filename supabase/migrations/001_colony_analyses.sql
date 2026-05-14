-- Colony Counter API: image retention table
-- Companion to the private `colony-uploads` Storage bucket.
-- Create the bucket separately (Supabase Dashboard → Storage → New bucket → Private)
-- or via the JS/CLI: supabase.storage.createBucket('colony-uploads', { public: false }).

create extension if not exists "pgcrypto";

create table if not exists public.colony_analyses (
    id              uuid        primary key default gen_random_uuid(),
    created_at      timestamptz not null    default now(),
    user_email      text,
    image_path      text        not null,
    colony_count    integer,
    density_channel text,
    image_metadata  jsonb
);

create index if not exists colony_analyses_created_at_idx
    on public.colony_analyses (created_at desc);

create index if not exists colony_analyses_user_email_idx
    on public.colony_analyses (user_email);
