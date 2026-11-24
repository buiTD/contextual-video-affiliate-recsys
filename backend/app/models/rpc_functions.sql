-- v1.0 RPCs: exact KNN for the retrieval parity test (N10) and atomic feedback
-- counters (N14). Profile EMA math stays in Python (TDD core, N13).

-- Cosine similarity against the enriched item vectors. filter_category and
-- exclude_category serve the two soft-retrieval arms (20 in-category, 10 global).
-- enable_indexscan=off keeps this exact KNN: the N10 parity contract must not
-- depend on whether the planner picks the HNSW index after an ANALYZE.
create or replace function public.match_products(
  query_emb extensions.vector(768),
  match_count integer default 30,
  filter_category text default null,
  exclude_category text default null
)
returns table (item_id text, name text, category text, price numeric, cosine real)
language sql
stable
set search_path = public, extensions
set enable_indexscan = off
as $$
  select p.id, p.name, p.category, p.price, (1 - (p.emb_ctx <=> query_emb))::real
  from public.products p
  where (filter_category is null or p.category = filter_category)
    and (exclude_category is null or p.category <> exclude_category)
  order by p.emb_ctx <=> query_emb
  limit least(coalesce(match_count, 30), 100);
$$;

-- Returns false when this (request, item, action) was already logged, so counters
-- never double-count. Raises when no matching impression exists: a click on an item
-- that was never shown would push CTR above 1 and poison position/OPE statistics.
-- The revoke below is what blocks anon; INVOKER keeps RLS as a second wall in case
-- that EXECUTE grant ever leaks (Supabase grants it to anon by default on new functions).
create or replace function public.record_feedback(
  p_request_id uuid,
  p_item_id text,
  p_action text,
  p_user_id uuid default null,
  p_script_id text default null,
  p_position smallint default null
)
returns boolean
language plpgsql
set search_path = public
as $$
declare
  inserted integer;
begin
  if not exists (
    select 1 from public.impressions
    where request_id = p_request_id and item_id = p_item_id
  ) then
    raise exception 'no impression for request % and item %', p_request_id, p_item_id;
  end if;

  insert into public.interaction_logs (request_id, user_id, script_id, item_id, action, position)
  values (p_request_id, p_user_id, p_script_id, p_item_id, p_action, p_position)
  on conflict (request_id, item_id, action) do nothing;

  get diagnostics inserted = row_count;
  if inserted = 0 then
    return false;
  end if;

  if p_action <> 'like' then
    update public.products set
      n_click = n_click + (p_action = 'click')::integer,
      n_dislike = n_dislike + (p_action = 'dislike')::integer
    where id = p_item_id;
  end if;

  return true;
end;
$$;

revoke execute on function public.record_feedback(uuid, text, text, uuid, text, smallint)
  from public, anon, authenticated;
grant execute on function public.record_feedback(uuid, text, text, uuid, text, smallint)
  to service_role;