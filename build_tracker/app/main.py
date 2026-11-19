from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import DEFAULT_MATCH_COUNT, FAKER, WEB_DIR
from app.db import export_all, get_match, get_meta, init_db, list_matches, list_players, summary
from app.ddragon import latest_version
from app.ingest import has_api_key, ingest_faker
from app.telemetry import ShopDecisionSessions


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Build Tracker", version="0.1.0", lifespan=lifespan)
# The public site (videns.gg, GitHub Pages) calls GET /api/summoner from the
# browser; the static origin is not this one. Read-only GETs only, no cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https://(www\.)?videns\.gg|https://javiops\.github\.io|https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
    allow_methods=["GET"],
    allow_headers=["*"],
    max_age=3600,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def faker_puuid() -> str | None:
    return get_meta("faker_puuid")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/live")
def live_page() -> FileResponse:
    return FileResponse(WEB_DIR / "live.html")


@app.get("/search")
def search_page() -> FileResponse:
    """Public landing: Riot ID search over the live Riot API."""
    return FileResponse(WEB_DIR / "search.html")


@app.get("/review")
def review_page() -> FileResponse:
    return FileResponse(WEB_DIR / "review.html")


@app.get("/review-faker-data.js")
def review_faker_data() -> FileResponse:
    return FileResponse(WEB_DIR / "review-faker-data.js", media_type="application/javascript")


@app.get("/review-playback.js")
def review_playback() -> FileResponse:
    return FileResponse(WEB_DIR / "review-playback.js", media_type="application/javascript")


@app.get("/review-faker-evidence.json")
def review_faker_evidence() -> FileResponse:
    return FileResponse(WEB_DIR / "review-faker-evidence.json", media_type="application/json")


@app.get("/api/review/{match_id}")
def api_review(
    match_id: str,
    region: str = "euw",
    participant_id: int | None = Query(None, ge=1, le=20),
) -> dict:
    from app.review import compile_review

    try:
        return compile_review(match_id, region, participant_id)
    except Exception as exc:
        from app.profile import ProfileError

        if isinstance(exc, ProfileError):
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        raise


@app.get("/api/summoner")
def api_summoner(
    riot_id: str = Query(..., min_length=3, max_length=40),
    region: str = "euw",
    count: int = Query(10, ge=1, le=20),
    refresh: bool = False,
) -> dict:
    from app.profile import ProfileError, compile_profile

    try:
        return compile_profile(riot_id, region, count, refresh=refresh)
    except ProfileError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/api/summoner/matches")
def api_summoner_matches(
    riot_id: str = Query(..., min_length=3, max_length=40),
    region: str = "euw",
    start: int = Query(20, ge=0, le=980),
    count: int = Query(20, ge=1, le=20),
) -> dict:
    """The page's Load more: the next slice of a player's match history."""
    from app.profile import ProfileError, compile_matches

    try:
        return compile_matches(riot_id, region, start, count)
    except ProfileError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/api/summoner/season")
def api_summoner_season(
    riot_id: str = Query(..., min_length=3, max_length=40),
    region: str = "euw",
    after: int = Query(0, ge=0),
) -> dict:
    """Season champion totals: starts a background sync and reports its progress."""
    from app.profile import ProfileError, season_status

    try:
        return season_status(riot_id, region, after)
    except ProfileError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/api/corpus")
def api_corpus() -> dict:
    from app.profile import REGIONS, corpus_totals

    return {
        "totals": corpus_totals(),
        "regions": [
            {"key": key, "label": region["label"]} for key, region in REGIONS.items()
        ],
    }


_predictor = None
_live_dragon = None
_shop_sessions = ShopDecisionSessions()


def _get_predictor():
    global _predictor
    if _predictor is None:
        from app.predictor import Predictor

        _predictor = Predictor()
    return _predictor


def _get_live_dragon():
    """Shape an idle live state without loading a model or emitting advice."""
    global _live_dragon
    if _live_dragon is None:
        from app.ddragon import default_dragon

        _live_dragon = default_dragon()
    return _live_dragon


def _live_payload(row: dict, dragon, session: dict | None = None, predictor=None) -> dict:
    """One response schema for idle and explicit-shop decision states.

    Crucially, options are absent outside a user-started shop session.  The
    local client API has no reliable location/shop flag, so guessing would put
    a shop-only model in the wrong serving population.
    """
    active = bool(session and session.get("active"))
    provenance = getattr(predictor, "provenance", None) if predictor else None
    return {
        "in_game": True,
        "decision_session": session,
        "advice_ready": active,
        "options": (session or {}).get("options", []) if active else [],
        "champion": row["champion"],
        "role": row["role"],
        "gold": row["gold"],
        "level": row["level"],
        "game_time_s": row["ts"] // 1000,
        "inventory": [{"item_id": i, "name": dragon.item_name(i)} for i in row["inventory"]],
        "ddragon_version": dragon.version,
        "model_trained_at": getattr(predictor, "trained_at", None),
        "model_provenance": provenance,
        # Legacy payload fields deliberately stay inert: third-party panels
        # cannot accidentally treat continuous live ranking as shop advice.
        "save": None,
        "top": [],
        "basket": [],
    }


@app.get("/api/live")
def api_live() -> dict:
    from app.live import fetch_snapshot, snapshot_to_row

    snap = fetch_snapshot()
    if not snap:
        return {"in_game": False}
    dragon = _get_live_dragon()
    row = snapshot_to_row(snap, dragon)
    if not row:
        return {"in_game": False}
    session = _shop_sessions.observe(row)
    return _live_payload(row, dragon, session=session, predictor=_predictor)


@app.post("/api/live/session")
def start_live_shop_session() -> dict:
    """Start an explicit, logged shop-decision session and compute advice once."""
    from app.live import fetch_snapshot, snapshot_to_row

    snap = fetch_snapshot()
    if not snap:
        raise HTTPException(status_code=409, detail="No game detected.")
    predictor = _get_predictor()
    row = snapshot_to_row(snap, predictor.dragon)
    if not row:
        raise HTTPException(status_code=409, detail="Could not identify the active player.")
    if not predictor.deployment.get("eligible"):
        raise HTTPException(
            status_code=409,
            detail=(
                "Advice is unavailable because the served model has not passed the "
                f"promotion gate: {predictor.deployment.get('reason', 'unknown reason')}."
            ),
        )
    # A hold card is only permitted inside this explicit session; the legacy
    # save target is post-death no-buy, not every possible recall.
    options = predictor.predict_options(row, budget_slack=0, allow_save=True)
    session = _shop_sessions.start(row, options, predictor.telemetry_metadata())
    return _live_payload(row, predictor.dragon, session=session, predictor=predictor)


@app.delete("/api/live/session")
def cancel_live_shop_session() -> dict:
    return {"cancelled": _shop_sessions.cancel()}


@app.get("/api/status")
def api_status(puuid: str | None = None) -> dict:
    data = summary(puuid)
    data["has_api_key"] = has_api_key()
    data["target"] = FAKER
    data["ddragon_version"] = latest_version()
    data["selected_puuid"] = puuid
    if puuid:
        data["player_games"] = len(list_matches(puuid))
    rank1 = get_meta("rank1_puuid")
    if rank1:
        data["featured"] = {
            "puuid": rank1,
            "label": get_meta("rank1_label") or "KR #1",
        }
    return data


@app.get("/api/players")
def api_players(q: str = "") -> dict:
    return {"players": list_players(q)}


@app.get("/api/sync")
def api_sync(count: int = Query(DEFAULT_MATCH_COUNT, ge=1, le=50)):
    if not has_api_key():
        raise HTTPException(
            status_code=400,
            detail="Add RIOT_API_KEY to a .env file, then restart the server.",
        )
    return StreamingResponse(
        _sync_stream(count),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sync_stream(count: int):
    q: queue.Queue = queue.Queue()

    def emit(event: dict) -> None:
        q.put(event)

    def run() -> None:
        try:
            ingest_faker(count=count, progress=emit, force=True)
        except Exception as exc:
            q.put({"step": "error", "message": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while True:
        event = q.get()
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"


@app.get("/api/matches")
def api_matches(puuid: str | None = None) -> dict:
    return {"matches": list_matches(puuid), "puuid": puuid}


@app.get("/api/matches/{match_id}")
def api_match(match_id: str, puuid: str | None = None) -> dict:
    match = get_match(match_id, puuid)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@app.get("/api/export")
def api_export() -> JSONResponse:
    payload = export_all(faker_puuid())
    return JSONResponse(
        payload,
        headers={"Content-Disposition": "attachment; filename=faker-ranked.json"},
    )
