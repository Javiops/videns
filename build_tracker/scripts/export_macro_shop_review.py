"""Export the reviewed jungler's shop visits for the macro Review (videns-site/macro.html).

The macro Review shows 40 challenger jungle games decision by decision. This
script adds the shop side of the reviewed player (the page's "POV"): what they
held, what they bought, the reconstructed wallet next to the HUD wallet, and --
only when explicitly asked for -- the options a mixture package displays at
that shop. It lives here, not in macro_coach, because the package, its pinned
runtime, the row builder and the legality check all live here.

Why each step is the way it is:

* **Served artifact by default.** `--advisor served` (the default) exports
  options only for what `app.deployment` binds for serving. Today nothing is
  bound, so it exits 3 and writes nothing. A package that is not promoted runs
  only with `--advisor package:<name> --allow-unpromoted`, and every file it
  writes carries `served: false` and the package's own provenance status.
* **Activation first, in a clean process.** `activate_runtime` REBINDS
  economic symbols in `baseline` and pins Data Dragon; activating after a
  trainer or policy import would leave this process pricing baskets with
  different economics than the package was built against, with nothing
  failing. So nothing but the standard library is imported before it.
* **Rows come from hash-bound generations, never from tracker.db.** The rows
  are the frozen ones the package's world was built on
  (`decision-start-footwear-16.18-v2` / `prequential-v3`). `verify_generation`
  hashes TRAIN (v4) and VAL (v3) only; test payloads are never opened, not even
  to hash. One byte-level pass per file keeps the reviewed player's lines;
  everything else is never parsed.
* **The trainer's own preprocessing.** Kept lines go through
  `train_basket_ab.stream_split` -- the same path `score_frozen_on_generation.py`
  uses -- and are zipped 1:1 with the raw lines (that recovers
  `participant_id`, which `stream_rows` drops).
* **Options, never scores.** Only `MixturePolicy.options()` is called; costs
  come from `purchase_sequence` at the row's own wallet. Probabilities, scores,
  group ids and any aggregate "agreement" figure are never computed into a
  published file, and the published files are checked for them before writing.
* **Fail closed per game.** A game without a row in the package's
  reconstruction, or whose reconstruction failed its source checks, is listed
  as unavailable with a reason code; a shop whose options cannot be decoded is
  kept with `options_status: "decode_error"`, never dropped.

The macro review files (`review-data/<mid>.json`) are read, never written.
Outputs go to `--out` (default `data/reports/macro_shop_review/<stamp>`);
`--publish <dir>` copies only the per-game files and `index.json`. The target
must be a separate folder named `shop` (never the review folder, never one
holding games.json), and only files carrying this exporter's schema are ever
replaced or removed there.

  .venv\\Scripts\\python.exe scripts\\export_macro_shop_review.py --advisor none --out <dir>
  .venv\\Scripts\\python.exe scripts\\export_macro_shop_review.py --advisor package:mixture-v4-m2-v1 --allow-unpromoted \\
      --out data\\reports\\macro_shop_review\\<stamp> --publish ..\\videns-site\\review-data\\shop
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "videns-shop-review-v1"
GENERATIONS = ROOT / "data" / "dataset_generations"
V4 = GENERATIONS / "16.18-reconstructible-development-v4"
V3 = GENERATIONS / "16.18-reconstructible-development-v3"
CORPUS_V4 = ROOT / "data" / "corpora" / "16.18-reconstructible-v4"
REVIEW_1618 = ROOT / "data" / "acceptance" / "16.18_reconstructible_20260923" / "inputs" / "static_review.json"
PACKAGES = ROOT / "data" / "packages"
DEFAULT_GAMES = ROOT.parent / "videns-site" / "review-data" / "games.json"
DEFAULT_OUT = ROOT / "data" / "reports" / "macro_shop_review"
DDRAGON = "16.18.1"
GOLD_INPUT = "prequential-v3"
SAVE_ITEM = 999999
# Array position carries the order; the key names the tier the page labels.
TIERS = ("first_line", "alternative", "other_legal_line")
HUD_WINDOW_S = (-1.0, 12.0)
UNBOUNDED_GOLD = 1e9          # legality of a purchase apart from its price
ENGINE_FILES = ("app/shop_econ.py", "app/conditional_policy.py", "scripts/basket_econ_cache.py")

# Nothing below may reach a published file: no scores, probabilities, group
# ids, agreement aggregates, and no "best"/"optimal" framing.
FORBIDDEN_KEY_PARTS = ("score", "prob", "group", "agree", "best", "optimal", "logit", "rank", "confidence")
FORBIDDEN_TEXT = ("best", "optimal")

EXIT_NOTHING_SERVED = 3
EXIT_INCOMPATIBLE_BINDING = 4
EXIT_CONTRACT = 5


# ----------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(value):
    """JSON number without float noise: ints stay ints."""
    value = float(value)
    return int(value) if value.is_integer() else round(value, 1)


def dump(obj, pretty=False) -> bytes:
    text = json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None,
                      separators=None if pretty else (",", ":"))
    return (text + "\n").encode("utf-8")


def fail(code: int, message: str):
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(code)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=Path, default=DEFAULT_GAMES,
                    help="macro review games.json (read only); its folder holds review-data/<mid>.json")
    ap.add_argument("--advisor", default="served",
                    help="none | served | package:<name>  (default: served)")
    ap.add_argument("--allow-unpromoted", action="store_true",
                    help="required with package:<name>; refused with anything else")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default data/reports/macro_shop_review/<YYYYMMDD-HHMM>)")
    ap.add_argument("--publish", type=Path, default=None,
                    help="copy the per-game files and index.json here (e.g. ..\\videns-site\\review-data\\shop)")
    args = ap.parse_args()
    mode = args.advisor
    if mode not in ("none", "served") and not re.fullmatch(r"package:[A-Za-z0-9][A-Za-z0-9._-]*", mode):
        ap.error("--advisor must be none, served or package:<name>")
    if mode.startswith("package:") and not args.allow_unpromoted:
        ap.error("package:<name> is not the served artifact; pass --allow-unpromoted to export it as a reference")
    if args.allow_unpromoted and not mode.startswith("package:"):
        ap.error("--allow-unpromoted only applies to package:<name>")
    return args


def check_publish_dest(publish: Path, review_dir: Path) -> Path:
    """Refuse a --publish target that could clobber the macro review itself.

    The per-game files share their names (EUW1_<id>.json) with the macro review
    files, so the target must be a folder named `shop` that is not the review
    folder and holds no games.json. Checked before any work is done.
    """
    dest = publish.resolve()
    if dest == review_dir or dest.name != "shop" or (dest / "games.json").exists() or dest in review_dir.parents:
        fail(2, f"refusing to publish into {dest}: the target must be a separate 'shop' folder "
             f"(e.g. {review_dir / 'shop'}), not the macro review folder")
    return dest


def is_our_file(path: Path) -> bool:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA


def load_games(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    games = {}
    for g in doc["games"]:
        mid = g["mid"]
        if mid in games or not re.fullmatch(r"[A-Z0-9]+_\d+", mid):
            raise ValueError(f"games.json: bad or duplicate mid {mid!r}")
        if type(g["pid"]) is not int or not 1 <= g["pid"] <= 10:
            raise ValueError(f"games.json: bad pid for {mid}")
        games[mid] = {"pid": g["pid"], "champ": g["champ"], "label": g.get("label"),
                      "team": g.get("team"), "patch": g.get("patch")}
    return games


# ----------------------------------------------------------------- advisor

def resolve_advisor(args) -> dict:
    """Decide what, if anything, is scored. Imports nothing heavy."""
    if args.advisor == "none":
        return {"mode": "none"}
    if args.advisor.startswith("package:"):
        name = args.advisor.split(":", 1)[1]
        if not (PACKAGES / name / "manifest.json").is_file():
            fail(2, f"no package named {name} under {PACKAGES}")
        return {"mode": "package", "package": name, "served": False}
    # served: only a deployment manifest binding a package digest can pass.
    from app.deployment import DEPLOYMENT_MANIFEST_PATH
    if not DEPLOYMENT_MANIFEST_PATH.exists():
        fail(EXIT_NOTHING_SERVED, "nothing is bound for serving; no options exported "
             f"({DEPLOYMENT_MANIFEST_PATH} does not exist)")
    binding = json.loads(DEPLOYMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    name, digest = binding.get("package"), binding.get("package_identity_sha256")
    if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) and digest:
        pid_path = PACKAGES / name / "package_id.json"
        if pid_path.is_file() and json.loads(pid_path.read_text(encoding="utf-8")).get("manifest_sha256") == digest:
            return {"mode": "package", "package": name, "served": True}
    fail(EXIT_INCOMPATIBLE_BINDING,
         "the served binding is not a package digest; a single Predictor artifact's row contract "
         "(legacy prequential-v2, unpinned static data) is not these footwear prequential-v3 rows; no options exported")


# ----------------------------------------------------------------- sources

def admission_games(path: Path, mids) -> dict:
    """Decode only the requested games from the (pretty-printed) v4 decision admission."""
    text = path.read_text(encoding="utf-8")
    start = text.find('\n  "games": {')
    if start < 0:
        raise ValueError("decision_admission.json has no games object")
    decoder = json.JSONDecoder()
    out = {}
    for mid in mids:
        key = f'\n    "{mid}": '
        i = text.find(key, start)
        if i < 0:
            continue
        game, _ = decoder.raw_decode(text, i + len(key))
        if not isinstance(game, dict) or type(game.get("admitted_rows")) is not int:
            raise ValueError(f"decision_admission.json: malformed entry for {mid}")
        out[mid] = game
    return out


def scan(path: Path, wanted: dict) -> tuple[list, Counter]:
    """One byte-level pass: keep the reviewed player's lines of the wanted games.

    wanted: {match_id: participant_id}. Returns (kept lines in file order,
    all-player line count per wanted game). Nothing else is parsed.
    """
    keys = {mid.encode("ascii"): pid for mid, pid in wanted.items()}
    mk, pk = b'"match_id":"', b'"participant_id":'
    kept, counts = [], Counter()
    with open(path, "rb") as fh:
        for line in fh:
            i = line.find(mk)
            if i < 0:
                if line.strip():
                    raise ValueError(f"{path.name}: a line without match_id")
                continue
            j = line.find(b'"', i + len(mk))
            mid = line[i + len(mk):j]
            pid = keys.get(mid)
            if pid is None:
                continue
            counts[mid.decode("ascii")] += 1
            k = line.find(pk)
            if k < 0:
                raise ValueError(f"{path.name}: a {mid.decode()} line without participant_id")
            k += len(pk)
            e = k
            while e < len(line) and 48 <= line[e] <= 57:
                e += 1
            if int(line[k:e]) == pid:
                kept.append(line if line.endswith(b"\n") else line + b"\n")
    return kept, counts


def hud_batches(review_file: Path) -> tuple[list[dict], str]:
    """Distinct `compras` batches (the POV's last shop) of one macro review game.

    Returns (batches, sha256 of the bytes read). A decision that catches a shop
    mid-visit shows a truncated batch with the same first `t` as the full one a
    later decision shows; only the longest of such prefix-related batches is kept.
    """
    data = review_file.read_bytes()
    doc = json.loads(data.decode("utf-8"))
    seen, out = set(), []
    for d in doc.get("decisions") or []:
        compras = d.get("compras") or []
        if not compras:
            continue
        key = tuple((c["t"], c["item"]) for c in compras)
        if key in seen:
            continue
        seen.add(key)
        entries = sorted(compras, key=lambda c: c["t"])
        out.append({"t": entries[0]["t"], "entries": entries,
                    "key": tuple((c["t"], c["item"]) for c in entries)})
    out.sort(key=lambda b: (b["t"], -len(b["entries"])))
    kept = []
    for b in out:
        if any(k["t"] == b["t"] and k["key"][:len(b["key"])] == b["key"] for k in kept):
            continue
        kept.append(b)
    return kept, hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------- per shop

def counts_of(label_counts) -> Counter:
    return Counter({int(k): int(v) for k, v in (label_counts or {}).items()})


def split_at_visits(batches: list[dict], visit_starts_s: list[float]) -> list[dict]:
    """Cut HUD batches at every model visit that starts inside them.

    The model opens a new shop visit after a 10 s idle gap; the HUD's "last
    shop" does not, so one batch can span two model visits. Uncut, the whole
    batch joins the first visit and the second gets no HUD reading. The two
    clocks agree to ~0.1 s, hence the 0.5 s tolerance on both sides.
    """
    out = []
    for b in batches:
        first, last = b["entries"][0]["t"], b["entries"][-1]["t"]
        cuts = sorted(s for s in visit_starts_s if first + 0.5 < s <= last + 0.5)
        parts, cur, ci = [], [], 0
        for e in b["entries"]:
            while ci < len(cuts) and e["t"] >= cuts[ci] - 0.5:
                if cur:
                    parts.append(cur)
                    cur = []
                ci += 1
            cur.append(e)
        if cur:
            parts.append(cur)
        for part in parts:
            out.append({"t": part[0]["t"], "entries": part,
                        "key": tuple((c["t"], c["item"]) for c in part)})
    return out


def match_hud(ts_ms: int, batches: list[dict]):
    t = ts_ms / 1000.0
    lo, hi = t + HUD_WINDOW_S[0], t + HUD_WINDOW_S[1]
    cands = [b for b in batches if lo <= b["t"] <= hi]
    return min(cands, key=lambda b: abs(b["t"] - t)) if cands else None


def history_entry(raw: dict, hud) -> dict:
    counts = counts_of(raw.get("label_counts"))
    if SAVE_ITEM in counts:
        if counts != Counter({SAVE_ITEM: 1}) or raw.get("save_kind") != "no_buy_death":
            raise ValueError(f"{raw['match_id']} ts {raw['ts']}: SAVE label outside a no_buy_death row")
        bought = {"kind": "save"}
    else:
        bought = {"kind": "buy", "items": {str(i): n for i, n in sorted(counts.items())}}
    first = hud["entries"][0] if hud else None
    held = [int(i["item_id"] if isinstance(i, dict) else i) for i in (raw.get("inventory") or [])]
    return {
        "ts_ms": int(raw["ts"]),
        "t": round(raw["ts"] / 1000.0, 1),
        "kind": raw.get("decision"),
        "level": raw.get("level"),
        "wallet": {"model_gold": int(round(float(raw["gold_est"]))),
                   "hud_gold": first["oro"] if first else None,
                   "hud_verified": first["verificado"] if first else None,
                   "hud_interval": first["intervalo"] if first else None},
        "held": held,
        "bought": bought,
        "bought_hud": ([{"t": c["t"], "item": c["item"], "gold": c["oro"], "interval": c["intervalo"],
                         "verified": c["verificado"]} for c in hud["entries"]] if hud else None),
    }


def option_entries(row, opts, wards, dragon, purchase_sequence):
    """options() output -> page options with real buy order and recursive costs."""
    out = []
    for i, opt in enumerate(opts[:len(TIERS)]):
        if opt["kind"] == "save":
            out.append({"tier": TIERS[i], "kind": "save", "items": []})
            continue
        action = tuple(sorted(int(k) for k, n in opt["items"].items() for _ in range(int(n))))
        if wards & set(action):
            return None, "a ward reached a displayed option"
        seq = purchase_sequence(action, row, float(row["gold_est"]), dragon)
        if seq is None:
            return None, f"purchase_sequence found no legal order for {action}"
        items = [{"item_id": int(s["item_id"]), "cost": num(s["cost"])} for s in seq]
        out.append({"tier": TIERS[i], "kind": "buy", "items": items, "total": num(sum(s["cost"] for s in seq))})
    return out, None


def compare_bought(raw, row, options, wards, dragon, purchase_sequence) -> dict:
    """How the purchase relates to the displayed options, for one shop.

    their_option: index of the option equal to the ward-stripped label, or None.
    A ward-only buy strips to the empty basket, which the package maps to its
    'no main purchase' option (the hold), so it matches the hold when one is
    shown; `their_option_note: "ward_only"` records that case either way.
    bought_legal_in_model_state: the ward-stripped basket has a legal buy order
    in the row's state at an unbounded wallet (slots, footwear, recipes).
    bought_fits_model_wallet: it also has one at the row's gold_est; None when
    it is illegal at any wallet, so "cost more than the wallet" is never claimed
    for a basket that fails for another reason.
    """
    counts = counts_of(raw.get("label_counts"))
    hold = next((i for i, o in enumerate(options) if o["kind"] == "save"), None)
    if SAVE_ITEM in counts:
        return {"their_option": hold, "their_option_note": None,
                "bought_fits_model_wallet": True, "bought_legal_in_model_state": True}
    stripped = Counter({i: n for i, n in counts.items() if i not in wards})
    if not stripped:
        return {"their_option": hold, "their_option_note": "ward_only",
                "bought_fits_model_wallet": True, "bought_legal_in_model_state": True}
    their = next((i for i, o in enumerate(options) if o["kind"] == "buy"
                  and Counter(it["item_id"] for it in o["items"]) == stripped), None)
    action = tuple(sorted(stripped.elements()))
    fits = purchase_sequence(action, row, float(row["gold_est"]), dragon) is not None
    legal = fits or purchase_sequence(action, row, UNBOUNDED_GOLD, dragon) is not None
    return {"their_option": their, "their_option_note": None,
            "bought_fits_model_wallet": fits if legal else None, "bought_legal_in_model_state": legal}


def assert_publishable(obj, where="") -> None:
    """Walk a published document; refuse any score-like key or 'best'/'optimal' text."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            low = str(k).lower()
            if any(p in low for p in FORBIDDEN_KEY_PARTS):
                fail(EXIT_CONTRACT, f"refusing to publish: forbidden key {k!r} at {where}")
            assert_publishable(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_publishable(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        low = obj.lower()
        if any(p in low for p in FORBIDDEN_TEXT):
            fail(EXIT_CONTRACT, f"refusing to publish: forbidden text {obj!r} at {where}")


# ----------------------------------------------------------------- main

def main():
    started = time.monotonic()
    args = parse_args()                                              # 1
    games = load_games(args.games)
    review_dir = args.games.resolve().parent
    dest = check_publish_dest(args.publish, review_dir) if args.publish else None
    advisor = resolve_advisor(args)

    pol = mp = None
    static_review_sha = None
    if advisor["mode"] == "package":                                 # 2: runtime before anything else
        import app.mixture_policy as mp
        pkg_dir = PACKAGES / advisor["package"]
        static_review_sha = sha256_file(REVIEW_1618)
        active = mp.activate_runtime(pkg_dir, REVIEW_1618)
        pol = mp.load_package(advisor["package"])
    purchase_sequence = None
    if pol is not None:                                              # 3
        import train_basket_ab as v1
        from app.conditional_policy import purchase_sequence
    from app.dataset_generations import verify_generation

    out = (args.out or DEFAULT_OUT / datetime.now().strftime("%Y%m%d-%H%M")).resolve()
    if (out / "report.json").exists():
        fail(2, f"{out} already holds an export; choose another --out")
    (out / "shop").mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()                                            # 4: hash-bound sources, no test payload
    m4 = verify_generation(V4, verify_splits=("train",))
    m3 = verify_generation(V3, verify_splits=("val",))
    verify_s = round(time.monotonic() - t0, 1)
    assign4 = json.loads((V4 / "split_assignment.json").read_text(encoding="utf-8"))
    assign3 = json.loads((V3 / "split_assignment.json").read_text(encoding="utf-8"))
    runtime4 = m4.get("footwear_runtime") or {}
    if pol is not None:
        pm = pol.manifest
        expected_recon = pm["footwear_runtime"]["reconstruction_version"]
        expected_gold = pm["provenance"]["gold_input"]
        wards = set(int(w) for w in pm["policy"]["wards_stripped"])
        trained_on_v4 = pm["generation"]["name"] == V4.name
        if expected_gold != GOLD_INPUT or pol.dragon.version != DDRAGON:
            fail(EXIT_CONTRACT, "the package's wallet or static data is not the 16.18 prequential-v3 world")
    else:
        expected_recon, expected_gold, wards, trained_on_v4 = runtime4["reconstruction_version"], GOLD_INPUT, {2055, 772043}, None

    admission_path = CORPUS_V4 / "decision_admission.json"           # 5: row source per game
    admission_sha = sha256_file(admission_path)
    if admission_sha != m4["files"]["decision_admission.json"]["sha256"]:
        fail(EXIT_CONTRACT, "corpus decision_admission.json differs from the one bound by the v4 generation")
    admitted = admission_games(admission_path, games)
    q4 = json.loads((V4 / "qualification.json").read_text(encoding="utf-8"))
    screening = json.loads((CORPUS_V4 / "screening_report.json").read_text(encoding="utf-8"))
    excluded = set(screening.get("excluded_match_ids") or [])

    plan = {}
    for mid, g in games.items():
        if assign4.get(mid) == "train":
            plan[mid] = {"source": "v4_train", "in_training": True}
        elif assign3.get(mid) == "val":
            plan[mid] = {"source": "v3_val", "in_training": False}
        elif mid in excluded:
            ev = CORPUS_V4 / "excluded_evidence" / f"{mid}.json"
            plan[mid] = {"status": "unavailable", "reason": "source_checks_failed",
                         "detail": json.loads(ev.read_text(encoding="utf-8")).get("reasons") if ev.is_file() else None}
        elif mid in admitted:
            plan[mid] = {"status": "unavailable", "reason": "no_row_in_model_reconstruction",
                         "detail": (q4.get("quarantined_match_ids") or {}).get(mid)}
        else:
            plan[mid] = {"status": "unavailable", "reason": "not_in_reconstructed_corpus", "detail": None}

    sources = {"v4_train": (V4, "train", V4 / "visits_train.jsonl", m4),
               "v3_val": (V3, "val", V3 / "visits_val.jsonl", m3)}
    kept_raw, source_report = {}, {}
    t0 = time.monotonic()
    for tag, (gdir, split, path, gm) in sources.items():             # 6: byte-level pass
        wanted = {mid: games[mid]["pid"] for mid, p in plan.items() if p.get("source") == tag}
        runtime_ok = (gm.get("footwear_runtime") == pol.manifest["footwear_runtime"]) if pol else None
        if pol is not None and wanted and not runtime_ok:
            fail(EXIT_CONTRACT, f"{tag}: the generation's footwear runtime is not the package's; no options exported")
        lines, all_counts = scan(path, wanted)
        raws = []
        for line in lines:
            raw = json.loads(line)
            mid = raw["match_id"]
            problems = [k for k, ok in (
                ("participant_id", raw.get("participant_id") == games[mid]["pid"]),
                ("reconstruction_version", raw.get("reconstruction_version") == expected_recon),
                ("gold_est_version", raw.get("gold_est_version") == expected_gold),
                ("ddragon_version", raw.get("ddragon_version") == DDRAGON),
                ("champion", raw.get("champion") == games[mid]["champ"]),
                ("gold_est", raw.get("gold_est") is not None)) if not ok]
            if problems:
                fail(EXIT_CONTRACT, f"{tag} {mid} ts {raw.get('ts')}: row contract broken: {problems}")
            raws.append(raw)
        raw_path = out / f"rows_raw_{tag}.jsonl"
        raw_path.write_bytes(b"".join(lines))
        kept_raw[tag] = (raw_path, raws)
        source_report[tag] = {
            "generation": gdir.name, "split": split,
            "generation_manifest_sha256": sha256_file(gdir / "generation_manifest.json"),
            "file": str(path), "file_sha256_verified": gm["files"][path.name]["sha256"],
            "footwear_runtime_matches_package": runtime_ok,
            "rows_raw": str(raw_path), "rows_raw_sha256": sha256_file(raw_path), "rows_kept": len(lines),
            "games": len(wanted)}
        for mid in wanted:
            povs = [r for r in raws if r["match_id"] == mid]
            ad = admitted.get(mid)
            ad_pov = sorted(d["ts"] for d in (ad or {}).get("admitted_decision_ids", []) if d["participant_id"] == games[mid]["pid"])
            plan[mid].update(rows=len(povs), all_player_rows=all_counts[mid],
                             admitted_rows=(ad or {}).get("admitted_rows"), admitted_pov_rows=len(ad_pov))
            same_rows = ad is not None and all_counts[mid] == ad["admitted_rows"] and sorted(r["ts"] for r in povs) == ad_pov
            if tag == "v3_val" and not same_rows:
                plan[mid].update(status="unavailable", reason="row_count_mismatch")
            elif not povs:
                plan[mid].update(status="unavailable", reason="no_row_in_model_reconstruction")
            else:
                plan[mid]["status"] = "available"
                plan[mid]["admission_rows_match"] = same_rows
    scan_s = round(time.monotonic() - t0, 1)

    rows_by_tag = {}
    for tag, (raw_path, raws) in kept_raw.items():                   # 7: trainer preprocessing, zipped 1:1
        if pol is None:
            rows_by_tag[tag] = list(zip(raws, [None] * len(raws)))
            continue
        rows = list(v1.stream_split(raw_path, set(), False, pol.champ_index, pol.dragon))
        if len(rows) != len(raws):
            fail(EXIT_CONTRACT, f"{tag}: preprocessing returned {len(rows)} rows for {len(raws)} raw lines")
        for raw, row in zip(raws, rows):
            if raw["match_id"] != row["match_id"] or raw["ts"] != row["ts"]:
                fail(EXIT_CONTRACT, f"{tag}: preprocessing reordered rows at {raw['match_id']} {raw['ts']}")
        rows_by_tag[tag] = list(zip(raws, rows))

    stamp = None
    if pol is not None:
        stamp = {"id": advisor["package"], "identity_sha256": pol.digest,
                 "status": pol.manifest["provenance"]["status"], "served": advisor["served"],
                 "ddragon_version": pol.dragon.version, "options_per_shop": pol.n_options,
                 "wards_excluded": sorted(wards), "wallet": f"gold_est ({expected_gold})"}

    t0 = time.monotonic()
    index_games, files, game_report, review_files = {}, {}, {}, {}
    status_counts, options_status_counts = Counter(), Counter()
    outside_wallet = illegal_bought = ward_only = hud_joined = hud_missing = hud_unjoined = 0
    for mid, g in games.items():                                     # 8-12
        p = plan[mid]
        if p.get("status") != "available":
            status_counts["unavailable"] += 1
            index_games[mid] = {"status": "unavailable", "reason": p["reason"]}
            files[mid] = {"schema": SCHEMA, "mid": mid, "pid": g["pid"], "champion": g["champ"],
                          "status": "unavailable", "reason": p["reason"], "advisor": stamp, "shops": []}
            game_report[mid] = {k: v for k, v in p.items()}
            continue
        status_counts["available"] += 1
        tag = p["source"]
        pairs = [(raw, row) for raw, row in rows_by_tag[tag] if raw["match_id"] == mid]
        batches, review_files[mid] = hud_batches(review_dir / f"{mid}.json")
        batches = split_at_visits(batches, [int(raw["ts"]) / 1000.0 for raw, _ in pairs])
        opts_all, game_error = None, None
        if pol is not None:
            try:
                opts_all = pol.options([row for _, row in pairs])                 # 8: options only
            except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed silently
                game_error = f"{type(exc).__name__}: {exc}"
        used, shops, errors = set(), [], []
        for j, (raw, row) in enumerate(pairs):
            hud = match_hud(int(raw["ts"]), batches)
            if hud:
                used.add(id(hud))
                hud_joined += 1
            else:
                hud_missing += 1
            entry = history_entry(raw, hud)
            if pol is not None:
                if game_error:
                    options, err = None, game_error
                else:
                    options, err = option_entries(row, opts_all[j], wards, pol.dragon, purchase_sequence)   # 9
                if options is None:
                    entry.update(bought_fits_model_wallet=None, bought_legal_in_model_state=None,
                                 options_status="decode_error", options=[], their_option=None,
                                 their_option_note=None)
                    errors.append({"ts_ms": entry["ts_ms"], "error": err})
                else:
                    cmp = compare_bought(raw, row, options, wards, pol.dragon, purchase_sequence)  # 10
                    entry.update(options_status="ok", options=options, **cmp)
                    outside_wallet += cmp["bought_fits_model_wallet"] is False
                    illegal_bought += cmp["bought_legal_in_model_state"] is False
                    ward_only += cmp["their_option_note"] == "ward_only"
                options_status_counts[entry["options_status"]] += 1
            shops.append(entry)
        unjoined = [b["t"] for b in batches if id(b) not in used]
        hud_unjoined += len(unjoined)
        in_training = (p["in_training"] if trained_on_v4 else None) if pol is not None else None
        files[mid] = {"schema": SCHEMA, "mid": mid, "pid": g["pid"], "champion": g["champ"],
                      "status": "available", "in_model_training_data": in_training, "advisor": stamp,
                      "row_source": {"generation": sources[tag][0].name, "split": sources[tag][1],
                                     "reconstruction_version": expected_recon, "gold_est_version": expected_gold},
                      "shops": shops}
        index_games[mid] = {"status": "available", "in_model_training_data": in_training, "shops": len(shops)}
        game_report[mid] = {**p, "shops": len(shops), "hud_batches": len(batches),
                            "hud_batches_without_row": unjoined, "decode_errors": errors}
    score_s = round(time.monotonic() - t0, 1)

    index = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "advisor": stamp, "games": index_games}
    assert_publishable(index, "index")
    for mid, doc in files.items():
        assert_publishable(doc, mid)
    written = []
    for mid, doc in files.items():
        path = out / "shop" / f"{mid}.json"
        path.write_bytes(dump(doc))
        written.append(path)
    (out / "shop" / "index.json").write_bytes(dump(index))
    written.append(out / "shop" / "index.json")

    published, removed = [], []
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
        names = {p.name for p in written}
        # Only files this exporter wrote are ever replaced or removed; anything
        # else at a colliding name stops the publish before a byte changes.
        ours = {f.name for f in dest.glob("*.json") if is_our_file(f)}
        foreign = sorted(n for n in names if (dest / n).exists() and n not in ours)
        if foreign:
            fail(2, f"refusing to publish into {dest}: {foreign[:5]} exist and are not {SCHEMA} files")
        for stale in sorted(dest.glob("*.json")):
            if stale.name not in names and stale.name in ours:
                stale.unlink()
                removed.append(stale.name)
        for path in written:
            shutil.copyfile(path, dest / path.name)
            published.append(str(dest / path.name))

    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv[1:], "mode": advisor["mode"], "advisor": stamp,
        "package_identity_sha256": pol.digest if pol else None,
        "runtime": (active.get("runtime_version") if pol else None),
        "static_review": ({"path": str(REVIEW_1618), "sha256": static_review_sha} if pol else None),
        "games_json": {"path": str(args.games.resolve()), "sha256": sha256_file(args.games)},
        "review_files": {"dir": str(review_dir), "sha256": review_files},
        "sources": source_report,
        "decision_admission": {"path": str(admission_path), "sha256": admission_sha},
        "screening_report_sha256": sha256_file(CORPUS_V4 / "screening_report.json"),
        "counts": {"games": len(games), "by_status": dict(status_counts),
                   "by_reason": dict(Counter(p["reason"] for p in plan.values() if p.get("status") != "available")),
                   "shops": sum(v.get("shops", 0) for v in index_games.values()),
                   "shops_by_source": {tag: sum(game_report[m]["shops"] for m in games
                                                if plan[m].get("status") == "available" and plan[m]["source"] == tag)
                                       for tag in sources},
                   "options_status": dict(options_status_counts),
                   "bought_outside_model_wallet": outside_wallet if pol else None,
                   "bought_illegal_in_model_state": illegal_bought if pol else None,
                   "bought_ward_only": ward_only if pol else None,
                   "hud_joined": hud_joined, "hud_missing": hud_missing, "hud_batches_without_row": hud_unjoined},
        "games": game_report,
        "code_sha256": {"scripts/export_macro_shop_review.py": sha256_file(Path(__file__)),
                        **{f: sha256_file(ROOT / f) for f in ENGINE_FILES}},
        "outputs": [str(p) for p in written], "published": published, "publish_removed_stale": removed,
        "timing_s": {"verify": verify_s, "scan": scan_s, "score": score_s,
                     "total": round(time.monotonic() - started, 1)},
        "never_exported": "scores, probabilities, group ids, agreement aggregates; options are imitation, not advice",
    }
    (out / "report.json").write_bytes(dump(report, pretty=True))
    print(json.dumps({"out": str(out), "mode": advisor["mode"], "counts": report["counts"],
                      "published": len(published), "timing_s": report["timing_s"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
