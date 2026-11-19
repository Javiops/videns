"""Cola de partidas de jungla de pros (LEC + Agurin) en challenger EUW.

Semillas (data/pros_seeds.json):
  1. riot_ids explícitos (lo más fiable; añade smurfs ahí), y
  2. barrido de las 300 cuentas challenger EUW: gameName con token exacto
     igual a un nombre de pro.

Solo cuentan partidas donde LA SEMILLA jugó JUNGLA, cola 420 (solo/duo),
plataforma EUW1, parche vigente, >= 15 min y con 2 junglas. Si los pros no
llegan al objetivo, --fill-challenger rellena con junglas challenger por LP
(y lo dice). Cada partida aceptada se pre-descarga (mv5_match + mv5_timeline)
al corpus y su meta.json registra el POV del pro.

Uso: python scripts/collect_pro_games.py [--games 40] [--patch 16.18|auto]
     [--per-seed-cap 8] [--fill-challenger]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS, DATA, QUEUE_FILE, QUEUE_RANKED_SOLO, ensure_dirs, game_dir
from app.riot import Riot
from app.segments import participants

SEEDS_FILE = DATA / "pros_seeds.json"
MIN_DURATION_S = 900


def queued_ids() -> set:
    ids = set()
    if QUEUE_FILE.exists():
        for line in open(QUEUE_FILE, encoding="utf-8"):
            ids.add(json.loads(line)["match_id"])
    if CORPUS.exists():
        ids |= {d.name for d in CORPUS.iterdir() if d.is_dir()}
    return ids


def resolve_seeds(riot: Riot, cfg: dict) -> list:
    """[{label, riot_id, puuid, src}] — explícitos primero, luego barrido."""
    seeds, seen = [], set()

    for rid in cfg.get("riot_ids", []):
        name, tag = rid.split("#", 1)
        acc = riot.account_by_riot_id(name, tag)
        if acc is None:
            print(f"  AVISO: {rid} no resuelve (¿renombrado?); ponlo al dia en "
                  f"{SEEDS_FILE.name}")
            continue
        if acc["puuid"] in seen:
            continue
        seen.add(acc["puuid"])
        seeds.append({"label": name, "riot_id": rid, "puuid": acc["puuid"],
                      "src": "riot_id"})
        print(f"  semilla explicita: {rid}")

    names = {n.lower() for n in cfg.get("names", [])}
    league = riot.challenger_league()
    entries = sorted(league["entries"], key=lambda e: -e["leaguePoints"])
    print(f"  barriendo {len(entries)} cuentas challenger "
          f"(~{len(entries) * 1.3 / 60:.0f} min sin cache)...")
    for i, e in enumerate(entries):
        acc = riot.account_by_puuid(e["puuid"])
        if i % 50 == 49:
            print(f"    {i + 1}/{len(entries)}", flush=True)
        if acc is None:
            continue
        gname = acc.get("gameName", "")
        tokens = {t.lower() for t in gname.split()}
        hit = tokens & names
        if not hit or acc["puuid"] in seen:
            continue
        seen.add(acc["puuid"])
        label = sorted(hit)[0]
        label = next(n for n in cfg["names"] if n.lower() == label)
        seeds.append({"label": label, "riot_id": f"{gname}#{acc.get('tagLine', '')}",
                      "puuid": acc["puuid"], "src": "challenger-scan",
                      "lp": e["leaguePoints"]})
        print(f"  match en challenger: {gname}#{acc.get('tagLine', '')} -> {label} "
              f"({e['leaguePoints']} LP)")
    return seeds


def detect_patch(riot: Riot, seeds: list) -> str:
    """Parche modal de las partidas mas recientes de las primeras semillas."""
    votes = Counter()
    for s in seeds[:4]:
        ids = riot.match_ids(s["puuid"], count=3, queue=QUEUE_RANKED_SOLO) or []
        for mid in ids:
            if not mid.startswith("EUW1_"):
                continue
            m = riot.match(mid)
            if m:
                v = m["info"]["gameVersion"].split(".")
                votes[f"{v[0]}.{v[1]}"] += 1
    if not votes:
        raise SystemExit("no puedo detectar el parche (semillas sin partidas EUW)")
    patch, n = votes.most_common(1)[0]
    print(f"  parche vigente detectado: {patch} ({n}/{sum(votes.values())} votos)")
    return patch


def seed_jungle_games(riot: Riot, seed: dict, patch: str, have: set,
                      cap: int, added_total: int, target: int, fh) -> int:
    """Encola hasta `cap` partidas de jungla de esta semilla. Devuelve n."""
    added = 0
    ids = []
    for start in (0, 100):
        page = riot.match_ids(seed["puuid"], count=100,
                              queue=QUEUE_RANKED_SOLO, start=start) or []
        ids += page
        if len(page) < 100:
            break
    for mid in ids:
        if added >= cap or added_total + added >= target:
            break
        if not mid.startswith("EUW1_") or mid in have:
            continue
        m = riot.match(mid)
        if m is None:
            continue
        info = m["info"]
        if not info["gameVersion"].startswith(patch + "."):
            v = info["gameVersion"].split(".")
            game_patch = (int(v[0]), int(v[1]))
            target = tuple(int(x) for x in patch.split("."))
            if game_patch > target:
                continue      # partida en parche mas nuevo: seguir bajando
            break             # ids por fecha: al llegar a un parche viejo, fin
        if info["gameDuration"] < MIN_DURATION_S:
            continue
        me = next((p for p in info["participants"]
                   if p["puuid"] == seed["puuid"]), None)
        if me is None or me["teamPosition"] != "JUNGLE":
            continue
        P = participants(m)
        jg = [(pid, v["champ"], v["team"]) for pid, v in P.items() if v["role"] == "JG"]
        if len(jg) != 2:
            continue
        row = {"match_id": mid, "patch": patch, "duration": info["gameDuration"],
               "junglers": [{"pid": p, "champ": c, "team": t} for p, c, t in jg],
               "seed": {"label": seed["label"], "riot_id": seed["riot_id"],
                        "pid": me["participantId"], "champ": me["championName"],
                        "src": seed["src"]},
               "added": time.strftime("%Y-%m-%d %H:%M")}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        have.add(mid)
        prefetch(riot, mid, m, row)
        added += 1
        print(f"  + {mid}  {seed['label']} ({me['championName']}) vs "
              f"{next(c for p, c, t in jg if p != me['participantId'])}")
    return added


def prefetch(riot: Riot, mid: str, match: dict, queue_row: dict) -> None:
    """Deja mv5_match/mv5_timeline y el meta con el POV del pro en el corpus."""
    g = game_dir(mid)
    g.mkdir(parents=True, exist_ok=True)
    mf = g / "mv5_match.json"
    if not mf.exists():
        mf.write_text(json.dumps(match), encoding="utf-8")
    tf = g / "mv5_timeline.json"
    if not tf.exists():
        tl = riot.timeline(mid)
        if tl:
            tf.write_text(json.dumps(tl), encoding="utf-8")
    meta_f = g / "meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    meta.update({"patch": queue_row["patch"], "seed": queue_row["seed"]})
    meta_f.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def fill_from_challenger(riot: Riot, patch: str, have: set, missing: int,
                         skip_puuids: set, fh) -> list:
    """Relleno: junglas challenger por LP (máx 4 partidas por jugador)."""
    league = riot.challenger_league()
    entries = sorted(league["entries"], key=lambda e: -e["leaguePoints"])
    filled = []
    for e in entries:
        if missing <= 0:
            break
        if e["puuid"] in skip_puuids:
            continue
        acc = riot.account_by_puuid(e["puuid"]) or {}
        label = f"challenger:{acc.get('gameName', e['puuid'][:8])}"
        seed = {"label": label, "riot_id":
                f"{acc.get('gameName', '?')}#{acc.get('tagLine', '?')}",
                "puuid": e["puuid"], "src": "challenger-fill"}
        n = seed_jungle_games(riot, seed, patch, have, cap=min(4, missing),
                              added_total=0, target=missing, fh=fh)
        if n:
            filled.append((label, n))
            missing -= n
    return filled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--patch", default="auto")
    ap.add_argument("--per-seed-cap", type=int, default=8)
    ap.add_argument("--fill-challenger", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    riot = Riot()
    cfg = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))

    print("1. resolviendo semillas...")
    seeds = resolve_seeds(riot, cfg)
    if not seeds:
        raise SystemExit("sin semillas: revisa data/pros_seeds.json")
    print(f"   {len(seeds)} semillas: "
          + ", ".join(f"{s['label']}({s['src'][0]})" for s in seeds))

    patch = detect_patch(riot, seeds) if args.patch == "auto" else args.patch

    have = queued_ids()
    per_seed = Counter()
    total = 0
    with open(QUEUE_FILE, "a", encoding="utf-8") as fh:
        print(f"2. partidas de jungla de pros (cap {args.per_seed_cap}/semilla)...")
        for s in seeds:
            if total >= args.games:
                break
            n = seed_jungle_games(riot, s, patch, have, args.per_seed_cap,
                                  total, args.games, fh)
            per_seed[s["label"]] += n
            total += n
        if total < args.games:
            print(f"3. segundo pase sin cap ({total}/{args.games})...")
            for s in seeds:
                if total >= args.games:
                    break
                n = seed_jungle_games(riot, s, patch, have, args.games - total,
                                      total, args.games, fh)
                per_seed[s["label"]] += n
                total += n
        if total < args.games and args.fill_challenger:
            print(f"4. relleno con junglas challenger ({total}/{args.games})...")
            filled = fill_from_challenger(riot, patch, have, args.games - total,
                                          {s["puuid"] for s in seeds}, fh)
            for label, n in filled:
                per_seed[label] += n
                total += n

    print(f"\nRESUMEN: {total}/{args.games} partidas encoladas (parche {patch})")
    for label, n in per_seed.most_common():
        if n:
            print(f"  {label:30s} {n}")
    if total < args.games:
        print("CORTO: los pros no dan mas este parche; opciones: "
              "--fill-challenger, ampliar riot_ids con smurfs, o esperar dias")
    print("\nSiguiente por partida: fetch_replay.py <ID> --watch -> "
          "capture_game.py <ID> -> process_game.py <ID>")
    print("Las replays caducan al cambiar el parche: descargarlas pronto.")


if __name__ == "__main__":
    main()
