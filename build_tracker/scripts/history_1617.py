"""History-corpus contract for the prior-patch transfer (patch 16.17): manifest + validator.

Owner's condition (2026-09-27): "formato distinto, mismas garantías de datos".
The history export is NOT a generation (no promotion path, no serving), but it
carries its own manifest and this validator, used both right after exporting
(`export_history_1617.py`) and before any training reads it
(`prepare_transfer_caches.py`, `basket_transfer_ab.py`). It refuses partial
results, digest mismatches, altered decision masks, changed row-builder code,
export code that differs from the audited build code, mixed versions,
duplicate rows, rows outside the audited decision ids, relabelled exclusions
(every export exclusion is recomputed from the frozen corpus) and any game
that ANY split file of the repository holds as `test` or `val` (infiltrated
holdouts). Every check is a comparison against frozen evidence: the qualified
corpus, its decision mask, its semantic audit, the completed run's protocol
and the frozen source plan.

The split-file scan (`split_files`) is shared with the snapshot, the build's
plan validation and the evidence verifier, so the four gates agree on what a
holdout is: fold `test`/`val` in any `split_assignment.json`,
`preserved_split_assignment.json` or `split_plan.json` under data/ (raw
archives and static caches excepted; they hold no split metadata).
"""
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "data" / "history" / "16.17-history-v1"
PURPOSE = "prior_patch_transfer_pretraining_only"
ROW_BUILDER_CODE = ("scripts/corpus_rows.py", "scripts/baseline.py", "app/shop_econ.py", "app/ddragon.py",
                    "app/purchase_multiset.py", "app/reconstruct.py", "app/footwear_reconstruction.py",
                    "scripts/footwear_economics.py", "scripts/footwear_timer.py", "scripts/footwear_timer_inventory.py",
                    "scripts/magical_footwear.py", "scripts/reconstructible_1617_runtime.py")
ROW_VERSION_FIELDS = ("reconstruction_version", "inventory_version", "gold_est_version", "slot_state_version",
                      "label_source_version", "footwear_state_version", "patch")
SPLIT_FILE_NAMES = ("split_assignment.json", "preserved_split_assignment.json", "split_plan.json")
SPLIT_SCAN_SKIP = ("raw_match_v5", "cache")          # raw archives and static data: no split metadata lives there
# The only exclusions corpus_rows.iter_games can emit, in its order of precedence.
EXPORT_EXCLUSION_REASONS = ("duration_below_600s", "incomplete_shop_perspectives", "no_purchase_or_save_targets")
MIN_DURATION_S = 600


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fail(msg):
    raise ValueError("history corpus rejected: " + msg)


# ---------------------------------------------------------------- split files (shared by the four gates)
def split_files(root=ROOT, exclude_dirs=()):
    """Every split file under data/ (any depth), except under the excluded directories (a snapshot's own outputs)."""
    base = (Path(root) / "data").resolve()
    excluded = [Path(p).resolve() for p in exclude_dirs]
    found = []
    for dirpath, dirnames, filenames in os.walk(base):
        here = Path(dirpath)
        if here == base:
            dirnames[:] = [d for d in dirnames if d not in SPLIT_SCAN_SKIP]
        if any(here == d or d in here.parents for d in excluded):
            dirnames[:] = []
            continue
        dirnames.sort()
        for name in filenames:
            if name in SPLIT_FILE_NAMES:
                found.append(here / name)
    return sorted(found)


def split_assignment(path):
    """The match_id -> fold mapping of a split file (a flat mapping, or a plan with an `assignment` mapping)."""
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("assignment"), dict):
        data = data["assignment"]
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"unrecognised split file layout: {path}")
    return data


def holdout_memberships(files):
    """(memberships mid -> {file: fold}, reserved test ids, historical validation holdout ids)."""
    memberships, reserved, holdouts = {}, set(), set()
    for path in files:
        for mid, fold in split_assignment(path).items():
            memberships.setdefault(mid, {})[str(path)] = fold
            if fold == "test":
                reserved.add(mid)
            elif fold == "val":
                holdouts.add(mid)
    return memberships, reserved, holdouts


# ---------------------------------------------------------------- export exclusions, recomputed from the corpus
def recompute_export_exclusion(conn, mid, admitted_rows):
    """The exclusion corpus_rows.iter_games emits for this qualified game, or None if it is exportable."""
    game = conn.execute("SELECT game_duration, participants_json FROM games WHERE match_id=?", (mid,)).fetchone()
    if game is None:
        fail(f"export-excluded game {mid} is not in the corpus")
    if game[0] < MIN_DURATION_S:
        return "duration_below_600s"
    participants = {p["puuid"] for p in json.loads(game[1])}
    shoppers = {json.loads(r[0])["puuid"] for r in conn.execute("SELECT payload_json FROM game_events WHERE match_id=?", (mid,))}
    if shoppers != participants:
        return "incomplete_shop_perspectives"
    if admitted_rows == 0:
        return "no_purchase_or_save_targets"
    return None


def check_export_exclusions(corpus, mask, exclusions):
    bad = {mid: r for mid, r in exclusions.items() if r not in EXPORT_EXCLUSION_REASONS}
    if bad:
        fail(f"export exclusions carry reasons the row builder cannot emit: {bad}")
    if not exclusions:
        return
    uri = (corpus / "tracker.db").as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        for mid, claimed in sorted(exclusions.items()):
            actual = recompute_export_exclusion(conn, mid, mask["games"][mid]["admitted_rows"])
            if actual != claimed:
                fail(f"export exclusion of {mid} does not hold on the frozen corpus (claimed {claimed}, recomputed {actual})")


# ---------------------------------------------------------------- the validator
def validate_history(directory=HISTORY_DIR, *, scan_rows=True, hash_db=True):
    """Return the manifest if every guarantee holds; raise ValueError otherwise."""
    directory = Path(directory).resolve()
    m = load_json(directory / "history_manifest.json")
    if (m.get("schema") != 1 or m.get("patch") != "16.17" or m.get("purpose") != PURPOSE
            or m.get("not_a_generation") is not True or m.get("promotion_allowed") is not False
            or m.get("test_payload_reads") is not False or m.get("state") != "complete"):
        fail("scope or state fields are not the declared ones (partial or foreign export)")
    corpus = Path(m["corpus"]["path"]).resolve()
    base = Path(m["acceptance"]["path"]).resolve()
    # ---- frozen evidence bindings ----
    cm_path = corpus / "corpus_manifest.json"
    if sha256_file(cm_path) != m["corpus"]["manifest_sha256"]:
        fail("corpus manifest digest changed")
    cm = load_json(cm_path)
    if cm.get("patch") != "16.17" or cm.get("db_sha256") != m["corpus"]["db_sha256"]:
        fail("corpus patch or database binding changed")
    if hash_db and sha256_file(corpus / "tracker.db") != m["corpus"]["db_sha256"]:
        fail("corpus database digest changed")
    mask_path = corpus / "decision_admission.json"
    if sha256_file(mask_path) != m["corpus"]["decision_admission_sha256"] or cm.get("decision_admission_sha256") != m["corpus"]["decision_admission_sha256"]:
        fail("decision mask altered")
    mask = load_json(mask_path)
    audit_path = base / "semantic_audit.json"
    if sha256_file(audit_path) != m["acceptance"]["semantic_audit_sha256"]:
        fail("semantic audit digest changed")
    audit = load_json(audit_path)
    if audit.get("passed") is not True or audit.get("hard_failures") or audit.get("quarantined_match_ids") \
            or audit.get("corpus_manifest_sha256") != m["corpus"]["manifest_sha256"] or audit.get("db_sha256") != m["corpus"]["db_sha256"]:
        fail("semantic audit does not certify this corpus")
    q_path = base / "qualification.json"
    if sha256_file(q_path) != m["acceptance"]["qualification_sha256"]:
        fail("qualification digest changed")
    q = load_json(q_path)
    if (q.get("status") != "qualified_reconstruction_snapshot" or q.get("patch") != "16.17"
            or q.get("corpus_manifest_sha256") != m["corpus"]["manifest_sha256"] or q.get("db_sha256") != m["corpus"]["db_sha256"]
            or q.get("decision_admission_sha256") != m["corpus"]["decision_admission_sha256"]):
        fail("qualification does not bind this corpus")
    plan_path = base / "source_plan.json"
    if sha256_file(plan_path) != m["acceptance"]["source_plan_sha256"] or cm.get("source_plan_sha256") != m["acceptance"]["source_plan_sha256"]:
        fail("source plan digest changed")
    plan = load_json(plan_path)
    if plan.get("patch") != "16.17" or plan.get("test_payload_reads") is not False or plan.get("training_or_export_authorized") is not False:
        fail("source plan scope changed")
    review_path = Path(plan["static_review"]).resolve()
    if sha256_file(review_path) != plan["static_review_sha256"] or sha256_file(review_path) != m["acceptance"]["static_review_sha256"]:
        fail("static review changed")
    if m["footwear_runtime"] != cm.get("footwear_runtime"):
        fail("runtime declaration differs from the corpus")
    # ---- the export ran the audited build code: protocol of the completed run vs the row-builder digests ----
    protocol_path = base / "run" / "protocol.json"
    if sha256_file(protocol_path) != m["acceptance"]["protocol_sha256"]:
        fail("completed-run protocol digest changed")
    protocol = load_json(protocol_path)
    if protocol.get("source_plan_sha256") != m["acceptance"]["source_plan_sha256"]:
        fail("completed-run protocol binds another source plan")
    build_code = protocol["code_sha256"]
    for rel in ROW_BUILDER_CODE:
        if rel in build_code and m["row_builder_code"].get(rel) != build_code[rel]:
            fail(f"the export did not run the audited build code: {rel}")
    if set(m["row_builder_code"]) != set(ROW_BUILDER_CODE):
        fail("row-builder code inventory differs from the contract")
    # ---- membership: exactly the qualified games; never a test or validation holdout anywhere ----
    accepted = set(q["accepted_match_ids"])
    games = set(m["games"]["exported"])
    excluded_export = m["games"]["export_exclusions"]
    if not games or games | set(excluded_export) != accepted or games & set(excluded_export):
        fail("exported + export-excluded games are not exactly the qualified games")
    if games != set(mask["games"]) - set(excluded_export):
        fail("exported games differ from the decision mask's games")
    if m["games"].get("qualified") != len(accepted) or m["games"].get("exported_count") != len(games):
        fail("game counts in the manifest do not match the membership")
    reserved = set(plan["reserved_test_ids"])
    holdouts = set(plan.get("historical_validation_holdout_ids", []))
    if games & reserved or games & holdouts:
        fail("a reserved test or historical holdout game was exported")
    memberships, live_reserved, live_holdouts = holdout_memberships(split_files(ROOT, exclude_dirs=[base]))
    infiltrated = games & (live_reserved | live_holdouts)
    if infiltrated:
        where = sorted({Path(f).relative_to(ROOT).as_posix() for mid in infiltrated for f, fold in memberships[mid].items() if fold in ("test", "val")})
        fail(f"{len(infiltrated)} exported games are test/val holdouts in {where[:3]}")
    digest = hashlib.sha256(("\n".join(sorted(games)) + "\n").encode("utf-8")).hexdigest()
    if digest != m["games"]["exported_sha256"]:
        fail("exported game membership digest changed")
    check_export_exclusions(corpus, mask, excluded_export)
    if m["games"].get("export_exclusion_counts") != dict(Counter(excluded_export.values())):
        fail("export exclusion counts do not match the listed exclusions")
    # ---- rows: complete, exact, unique, single-version, exactly the audited decision ids ----
    rows_path = directory / m["rows"]["file"]
    if not rows_path.is_file() or rows_path.stat().st_size != m["rows"]["bytes"] or sha256_file(rows_path) != m["rows"]["sha256"]:
        fail("row file missing, resized or digest changed")
    expected_per_game = {}
    admitted_keys = {}
    for mid in games:
        entry = mask["games"][mid]
        keys = {(d["participant_id"], d["ts"]) for d in entry["admitted_decision_ids"]}
        if len(keys) != len(entry["admitted_decision_ids"]) or len(keys) != entry["admitted_rows"]:
            fail(f"decision mask of {mid} is internally inconsistent")
        expected_per_game[mid] = entry["admitted_rows"]
        admitted_keys[mid] = keys
    if sum(expected_per_game.values()) != m["rows"]["count"]:
        fail("row count differs from the decision mask's admitted rows")
    for rel, digest in m["row_builder_code"].items():
        if sha256_file(ROOT / rel) != digest:
            fail(f"row-builder code changed since export: {rel}")
    if scan_rows:
        per_game, seen, versions, n = Counter(), defaultdict(set), {f: Counter() for f in ROW_VERSION_FIELDS}, 0
        with open(rows_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                n += 1
                key = (row["participant_id"], row["ts"])
                mid = row["match_id"]
                if mid not in admitted_keys:
                    fail(f"row of a non-exported game {mid}")
                if key in seen[mid]:
                    fail(f"duplicate row {(mid,) + key}")
                if key not in admitted_keys[mid]:
                    fail(f"row {(mid,) + key} is not an audited admitted decision")
                seen[mid].add(key)
                per_game[mid] += 1
                for f in ROW_VERSION_FIELDS:
                    versions[f][row.get(f)] += 1
        if n != m["rows"]["count"]:
            fail(f"row file holds {n} rows, manifest says {m['rows']['count']}")
        if dict(per_game) != expected_per_game:
            bad = [mid for mid in set(per_game) | set(expected_per_game) if per_game.get(mid) != expected_per_game.get(mid)]
            fail(f"per-game row counts differ from the decision mask for {len(bad)} games (e.g. {bad[:3]})")
        missing = [mid for mid, keys in admitted_keys.items() if seen[mid] != keys]
        if missing:
            fail(f"{len(missing)} games do not carry exactly their audited admitted decisions (e.g. {missing[:3]})")
        for f in ROW_VERSION_FIELDS:
            if dict(versions[f]) != {m["versions"][f]: n}:
                fail(f"rows carry mixed or unexpected {f}: {dict(versions[f])}")
    return m
