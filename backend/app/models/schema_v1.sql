-- v1.0 data model: 7 tables, RLS, pgvector HNSW.
-- Apply with the Supabase SQL editor. Functions, policies, indexes and the ALTER
-- below re-apply safely.
-- shortcut: `create table if not exists` never alters an existing table, so a new
-- column/constraint inside a CREATE block will NOT reach a live DB — add an explicit
-- ALTER (as done for interaction_logs.position) or drop/recreate the table.
-- Vectors are L2-normalized upstream (pipelines/embedder.py), so cosine == dot product.

create extension if not exists vector with schema extensions;

-- 1. Catalog. Counters feed the Bayes penalties (plan 4.3); they are bumped by
--    the impressions trigger and record_feedback, never by sync_db.
create table if not exists public.products (
  id text primary key,
  name text not null,
  category text not null
    check (category in ('book', 'tech', 'cosmetics', 'health', 'perfume', 'fashion')),
  price numeric(10, 2) not null check (price > 0),
  description text not null default '',
  target_personas jsonb not null default '[]'::jsonb,
  usage_scenarios jsonb not null default '[]'::jsonb,
  solved_problems jsonb not null default '[]'::jsonb,
  emb_raw extensions.vector(768) not null,
  emb_ctx extensions.vector(768) not null,
  emb_centered extensions.vector(768) not null,
  mean_version integer not null default 1,
  n_impr integer not null default 0,
  n_click integer not null default 0,
  n_dislike integer not null default 0
);

create index if not exists products_category_idx on public.products (category);
create index if not exists products_emb_ctx_hnsw
  on public.products using hnsw (emb_ctx extensions.vector_cosine_ops);

-- 2. Scripts. Latent columns are nullable: only the 50 feed scripts are enriched,
--    the 40 golden (dev/test) scripts still await enrichment.
create table if not exists public.scripts (
  id text primary key,
  split text not null check (split in ('feed', 'dev', 'test')),
  category text not null
    check (category in ('book', 'tech', 'cosmetics', 'health', 'perfume', 'fashion')),
  hook text not null,
  scenes jsonb not null,
  voiceover text not null,
  visual_cue text not null,
  emb_query extensions.vector(768) not null,
  dominant_mood text,
  latent_needs jsonb,
  recommended_categories jsonb,
  bridge_copy text
);

create index if not exists scripts_split_idx on public.scripts (split);

-- 3. Dual-vector EMA profile (plan 4.5). Vectors live in the mean-centered space
--    identified by mean_version, so a catalog rebuild can replay them.
create table if not exists public.user_profiles (
  user_id uuid primary key references auth.users (id) on delete cascade,
  pos_vec extensions.vector(768),
  neg_vec extensions.vector(768),
  mean_version integer not null default 1,
  n_events integer not null default 0,
  updated_at timestamptz not null default now()
);

-- 4. Serving log: one row per (request, item) shown, with the logging propensity.
create table if not exists public.impressions (
  id bigint generated always as identity primary key,
  request_id uuid not null,
  user_id uuid references auth.users (id) on delete set null,
  script_id text not null references public.scripts (id),
  item_id text not null references public.products (id),
  position smallint not null check (position >= 1),
  propensity real not null check (propensity > 0 and propensity <= 1),
  created_at timestamptz not null default now(),
  unique (request_id, item_id)
);

create index if not exists impressions_item_idx on public.impressions (item_id);

-- 5. Append-only event source: replaying it rebuilds profiles under a new mean.
--    The unique key is what makes feedback idempotent.
create table if not exists public.interaction_logs (
  id bigint generated always as identity primary key,
  request_id uuid not null,
  user_id uuid references auth.users (id) on delete set null,
  script_id text references public.scripts (id),
  item_id text not null references public.products (id),
  action text not null check (action in ('like', 'click', 'dislike')),
  position smallint check (position >= 1),
  created_at timestamptz not null default now(),
  unique (request_id, item_id, action)
);

-- The table already exists in live projects, so the CREATE above cannot add this.
alter table public.interaction_logs
  drop constraint if exists interaction_logs_position_check,
  add constraint interaction_logs_position_check check (position >= 1);

create index if not exists interaction_logs_user_time_idx
  on public.interaction_logs (user_id, created_at);

-- 6. Semantic cache entries (N22). Request-level cache state lives in telemetry.
create table if not exists public.semantic_cache (
  query_hash text primary key,
  query_text text not null,
  emb_query extensions.vector(768),
  payload jsonb not null,
  hits integer not null default 0,
  created_at timestamptz not null default now()
);

-- 7. Per-request telemetry (N15).
create table if not exists public.request_telemetry (
  request_id uuid primary key,
  endpoint text not null,
  user_id uuid,
  stage_ms jsonb not null default '{}'::jsonb,
  cold_start boolean not null default false,
  cache_state text check (cache_state in ('hit_exact', 'hit_semantic', 'miss')),
  thinking_tokens integer not null default 0,
  shadow_cost_usd numeric(10, 6) not null default 0,
  created_at timestamptz not null default now()
);

-- RLS: catalog is public-read, a user reads only their own rows, and all writes
-- go through the backend with service_role (BYPASSRLS). Cache and telemetry have
-- no policy at all, so anon/authenticated see nothing.
alter table public.products enable row level security;
alter table public.scripts enable row level security;
alter table public.user_profiles enable row level security;
alter table public.impressions enable row level security;
alter table public.interaction_logs enable row level security;
alter table public.semantic_cache enable row level security;
alter table public.request_telemetry enable row level security;

drop policy if exists products_read on public.products;
create policy products_read on public.products
  for select to anon, authenticated using (true);

drop policy if exists scripts_read on public.scripts;
create policy scripts_read on public.scripts
  for select to anon, authenticated using (true);

drop policy if exists own_profile on public.user_profiles;
create policy own_profile on public.user_profiles
  for select to authenticated using (auth.uid() = user_id);

drop policy if exists own_impressions on public.impressions;
create policy own_impressions on public.impressions
  for select to authenticated using (auth.uid() = user_id);

drop policy if exists own_logs on public.interaction_logs;
create policy own_logs on public.interaction_logs
  for select to authenticated using (auth.uid() = user_id);

grant select on public.products, public.scripts to anon, authenticated;
grant select on public.user_profiles, public.impressions, public.interaction_logs
  to authenticated;
grant all on public.products, public.scripts, public.user_profiles, public.impressions,
  public.interaction_logs, public.semantic_cache, public.request_telemetry to service_role;
grant usage, select on all sequences in schema public to service_role;

create or replace function public.bump_impression_count()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  update public.products set n_impr = n_impr + 1 where id = new.item_id;
  return new;
end;
$$;

drop trigger if exists impressions_bump_count on public.impressions;
create trigger impressions_bump_count
  after insert on public.impressions
  for each row execute function public.bump_impression_count();