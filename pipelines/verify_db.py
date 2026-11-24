"""Live verification of schema_v1.sql + rpc_functions.sql against a real Supabase project.

Not collected by pytest: it writes to the database. It creates scratch impressions,
feedback rows and one probe auth user, then deletes them all and resets the counters
it touched. Run it after applying any SQL change, from backend/:

    NO_PROXY=localhost,127.0.0.1 python -m uv run --env-file ../.env python -m pipelines.verify_db

Needs SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_ANON_KEY and the local
data/raw + data/processed artifacts. Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import truststore
from supabase import create_client

from pipelines.crawlers.product_loader import load_and_validate_products
from pipelines.crawlers.script_loader import load_curated_scripts
from pipelines.embedder import DIM
from pipelines.sync_db import main as sync_main

truststore.inject_into_ssl()  # corporate MITM CA lives in the OS store, not certifi

ROOT = Path(__file__).resolve().parents[1]
PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(label)
    print(("PASS " if cond else "FAIL ") + label + (f" | {detail}" if detail else ""))


def denied(label: str, fn: object) -> None:
    """A read is blocked either by GRANT (42501) or by RLS returning no rows."""
    try:
        rows = fn()  # type: ignore[operator]
        check(label, rows == [], f"returned {rows}")
    except Exception as exc:
        check(label, "42501" in str(exc), str(exc)[:100])


def raises(label: str, fn: object, needle: str) -> None:
    try:
        fn()  # type: ignore[operator]
        check(label, False, "call succeeded")
    except Exception as exc:
        check(label, needle in str(exc), str(exc)[:100])


def main() -> int:
    url = os.environ["SUPABASE_URL"]
    svc = create_client(url, os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    anon = create_client(url, os.environ["SUPABASE_ANON_KEY"])
    proc = ROOT / "data" / "processed"
    products = load_and_validate_products(ROOT / "data" / "raw" / "products_raw.json")
    scripts = load_curated_scripts(ROOT / "data" / "raw" / "scripts_raw.json")
    ids = [p.id for p in products]
    cats = np.asarray([p.category for p in products])
    emb_ctx = np.load(proc / "product_emb_ctx.npy")
    script_emb = np.load(proc / "script_emb.npy")

    def snapshot() -> str:
        rows = svc.table("products").select("id,name,price,category,mean_version").execute().data
        return json.dumps(sorted(rows, key=lambda r: r["id"]), sort_keys=True)

    def counters() -> tuple:
        rows = svc.table("products").select("n_impr,n_click,n_dislike").execute().data
        return tuple(sum(r[k] for r in rows) for k in ("n_impr", "n_click", "n_dislike"))

    def rpc(args: dict) -> list:
        return svc.rpc("match_products", args).execute().data

    print("== catalog ==")
    snap = snapshot()
    n_db = len(json.loads(snap))
    db_scripts = svc.table("scripts").select("id,split,dominant_mood").execute().data
    splits = {s: sum(1 for r in db_scripts if r["split"] == s) for s in ("feed", "dev", "test")}
    nulls = sum(1 for r in db_scripts if r["split"] != "feed" and r["dominant_mood"] is None)
    check("products match local catalog", n_db == len(ids), str(n_db))
    check("scripts match local corpus", len(db_scripts) == len(scripts), str(len(db_scripts)))
    check("splits 50/24/16", splits == {"feed": 50, "dev": 24, "test": 16}, str(splits))
    check("unenriched goldens are NULL", nulls == 40, str(nulls))
    check("counters start clean", counters() == (0, 0, 0), str(counters()))

    print("== vector storage ==")
    one = svc.table("products").select("id,emb_ctx").limit(1).execute().data[0]
    stored = np.asarray(json.loads(one["emb_ctx"]), dtype=np.float32)
    local = emb_ctx[ids.index(one["id"])].astype(np.float32)
    check("dim 768", stored.shape == (DIM,))
    check("unit norm", abs(float(np.linalg.norm(stored)) - 1) < 1e-5)
    check("float32 round-trip is bit-exact", np.array_equal(stored, local))

    print("== RLS (anon) ==")
    n_anon = anon.table("products").select("id", count="exact").limit(1).execute().count
    check("anon reads catalog", n_anon == len(ids), str(n_anon))
    denied(
        "anon blocked from logs",
        lambda: anon.table("interaction_logs").select("id").execute().data,
    )
    denied(
        "anon blocked from telemetry",
        lambda: anon.table("request_telemetry").select("request_id").execute().data,
    )
    denied(
        "anon blocked from cache",
        lambda: anon.table("semantic_cache").select("query_hash").execute().data,
    )
    denied(
        "anon insert denied",
        lambda: (
            anon.table("interaction_logs")
            .insert({"request_id": str(uuid.uuid4()), "item_id": one["id"], "action": "click"})
            .execute()
            .data
        ),
    )
    denied(
        "anon record_feedback denied",
        lambda: (
            anon.rpc(
                "record_feedback",
                {"p_request_id": str(uuid.uuid4()), "p_item_id": one["id"], "p_action": "click"},
            )
            .execute()
            .data
        ),
    )

    print("== match_products bounds and parity ==")
    feed = [i for i, s in enumerate(scripts) if s.split == "feed"][:3]
    lit0 = json.dumps(script_emb[feed[0]].astype(np.float32).tolist())
    check(
        "null match_count falls back to 30",
        len(rpc({"query_emb": lit0, "match_count": None})) == 30,
    )
    check(
        "huge match_count capped at 100",
        len(rpc({"query_emb": lit0, "match_count": 100000})) == 100,
    )
    for i in feed:
        q = script_emb[i]
        lit = json.dumps(q.astype(np.float32).tolist())
        sims = emb_ctx @ q
        top = np.argsort(-sims)[:30]
        got = rpc({"query_emb": lit, "match_count": 30})
        gids = [r["item_id"] for r in got]
        diff = float(np.max(np.abs(np.asarray([r["cosine"] for r in got]) - sims[top])))
        sid = scripts[i].id
        check(f"top30 exact order {sid}", gids == [ids[j] for j in top], f"maxdiff={diff:.2e}")
        check(f"cosine within float32 tolerance {sid}", diff < 1e-5, f"{diff:.2e}")
        cat = scripts[i].category
        arm_in = rpc({"query_emb": lit, "match_count": 20, "filter_category": cat})
        idx = np.flatnonzero(cats == cat)
        want_in = [ids[idx[j]] for j in np.argsort(-sims[idx])[:20]]
        check(
            f"in-category arm 20 {cat}", [r["item_id"] for r in arm_in] == want_in, str(len(arm_in))
        )
        arm_out = rpc({"query_emb": lit, "match_count": 10, "exclude_category": cat})
        idx = np.flatnonzero(cats != cat)
        want_out = [ids[idx[j]] for j in np.argsort(-sims[idx])[:10]]
        check(
            f"global arm 10 ex-{cat}",
            [r["item_id"] for r in arm_out] == want_out,
            str(len(arm_out)),
        )

    print("== impressions, feedback, guards ==")
    req, item = str(uuid.uuid4()), one["id"]
    script_id = next(r["id"] for r in db_scripts if r["split"] == "feed")
    svc.table("impressions").insert(
        {
            "request_id": req,
            "script_id": script_id,
            "item_id": item,
            "position": 1,
            "propensity": 1.0,
        }
    ).execute()

    def item_counters() -> dict:
        row = svc.table("products").select("n_impr,n_click,n_dislike").eq("id", item).single()
        return row.execute().data

    check("trigger bumps n_impr", item_counters()["n_impr"] == 1, str(item_counters()))
    args = {"p_request_id": req, "p_item_id": item, "p_action": "click", "p_position": 1}
    check("first feedback true", svc.rpc("record_feedback", args).execute().data is True)
    check("replayed feedback false", svc.rpc("record_feedback", args).execute().data is False)
    check("click not double-counted", item_counters()["n_click"] == 1, str(item_counters()))
    svc.rpc("record_feedback", {**args, "p_action": "dislike"}).execute()
    check("dislike counted", item_counters()["n_dislike"] == 1, str(item_counters()))
    n_logs = len(svc.table("interaction_logs").select("id").eq("request_id", req).execute().data)
    check("3 calls produce 2 log rows", n_logs == 2, str(n_logs))
    raises(
        "feedback without impression rejected",
        lambda: svc.rpc(
            "record_feedback",
            {"p_request_id": str(uuid.uuid4()), "p_item_id": item, "p_action": "click"},
        ).execute(),
        "no impression",
    )
    raises(
        "position 0 rejected",
        lambda: (
            svc.table("interaction_logs")
            .insert({"request_id": req, "item_id": item, "action": "like", "position": 0})
            .execute()
        ),
        "23514",
    )
    before = counters()

    print("== RLS own-row policy (probe user) ==")
    email = f"n6-probe-{uuid.uuid4().hex[:8]}@example.com"
    pw = uuid.uuid4().hex + "Aa1!"
    created = svc.auth.admin.create_user({"email": email, "password": pw, "email_confirm": True})
    probe = created.user.id
    session = anon.auth.sign_in_with_password({"email": email, "password": pw})
    user_client = create_client(url, os.environ["SUPABASE_ANON_KEY"])
    user_client.postgrest.auth(session.session.access_token)
    seen = user_client.table("interaction_logs").select("id").execute().data
    check("probe cannot see NULL-owner logs", seen == [], str(seen))
    own_req = str(uuid.uuid4())
    svc.table("interaction_logs").insert(
        {"request_id": own_req, "user_id": probe, "item_id": item, "action": "like"}
    ).execute()
    svc.table("user_profiles").insert({"user_id": probe, "mean_version": 1}).execute()
    own = user_client.table("interaction_logs").select("user_id").execute().data
    check("probe sees only its own log", own == [{"user_id": probe}], str(own))
    prof = user_client.table("user_profiles").select("user_id").execute().data
    check("probe sees only its own profile", prof == [{"user_id": probe}], str(prof))

    print("== sync idempotency ==")
    sync_main(ROOT)
    check("no duplicate products", len(json.loads(snapshot())) == len(ids))
    check("catalog content unchanged", snapshot() == snap)
    check("counters survive re-sync", counters() == before, f"{before} -> {counters()}")
    n_scripts = svc.table("scripts").select("id", count="exact").execute().count
    check("no duplicate scripts", n_scripts == len(scripts), str(n_scripts))

    print("== cleanup ==")
    svc.table("interaction_logs").delete().in_("request_id", [req, own_req]).execute()
    svc.table("impressions").delete().eq("request_id", req).execute()
    svc.table("products").update({"n_impr": 0, "n_click": 0, "n_dislike": 0}).eq(
        "id", item
    ).execute()
    svc.auth.admin.delete_user(probe)
    check("counters reset", counters() == (0, 0, 0), str(counters()))
    check("logs empty", svc.table("interaction_logs").select("id").execute().data == [])
    check("impressions empty", svc.table("impressions").select("id").execute().data == [])
    check("profile cascaded", svc.table("user_profiles").select("user_id").execute().data == [])
    check("catalog intact", snapshot() == snap)

    print(f"\n{len(PASSED)} PASS / {len(FAILED)} FAIL")
    for label in FAILED:
        print("FAILED:", label)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
