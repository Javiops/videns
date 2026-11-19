from app import lexicon


def seg(kind, t0=100.0, t1=130.0, **kw):
    base = {"kind": kind, "t0": t0, "t1": t1, "fight_lanes": []}
    base.update(kw)
    return base


def test_basic_tokens():
    assert lexicon.token_for_segment(seg("reset")) == "RESET"
    assert lexicon.token_for_segment(seg("base")) is None
    assert lexicon.token_for_segment(seg("lane_gank", lane="MID")) == "GANK_MID"
    assert lexicon.token_for_segment(seg("lane_fight", lane="TOP")) == "PELEAR_TOP"
    assert lexicon.token_for_segment(seg("lane_cover", lane="BOT")) == "CUBRIR_BOT"
    assert lexicon.token_for_segment(
        seg("jungle", region_team="JGL_PA", farming=True)) == "FARMEAR_PROPIA_AZUL"
    assert lexicon.token_for_segment(
        seg("jungle", region_team="JGL_PA", farming=False)) is None  # cruza = tránsito
    assert lexicon.token_for_segment(
        seg("jungle", region_team="JGL_ER", farming=False)) == "INVADIR_ROJA"
    assert lexicon.token_for_segment(
        seg("river", region_team="RIO_TOP", farming=True)) == "CENTINELA_TOP"
    assert lexicon.token_for_segment(
        seg("river", region_team="RIO_BOT", farming=False)) == "ROTAR_RIO_BOT"


def test_jungle_fight_with_laner_is_gank():
    # regla auditada fase 1: GANK TOP -> INVADE ...
    s = seg("jungle", region_team="JGL_ER", farming=True, fight_lanes=["TOP"])
    assert lexicon.token_for_segment(s) == "GANK_TOP"


def test_baron_pit_clock_disambiguation():
    assert lexicon.token_for_segment(seg("pit_baron", t0=500)) == "OBJETIVO_LARVAS"
    assert lexicon.token_for_segment(seg("pit_baron", t0=1000)) == "OBJETIVO_HERALDO"
    assert lexicon.token_for_segment(seg("pit_baron", t0=1500)) == "OBJETIVO_BARON"
    # el evento manda sobre el reloj
    assert lexicon.token_for_segment(
        seg("pit_baron", t0=1500, monster="RIFTHERALD")) == "OBJETIVO_HERALDO"
    assert lexicon.token_for_segment(seg("pit_dragon", t0=800)) == "OBJETIVO_DRAGON"


def test_transit_inherits_forward():
    segs = [seg("jungle", 40, 70, region_team="JGL_PR", farming=True),
            seg("jungle", 75, 90, region_team="JGL_PA", farming=False),  # cruza
            seg("lane_gank", 95, 110, lane="MID")]
    d = lexicon.decisions_from_segments(segs)
    assert [x["token"] for x in d] == ["FARMEAR_PROPIA_ROJA", "GANK_MID", "GANK_MID"][:2] \
        or [x["token"] for x in d] == ["FARMEAR_PROPIA_ROJA", "GANK_MID"]
    # el tránsito hereda GANK_MID y se colapsa con el gank real
    assert d[1]["t0"] == 75


def test_consecutive_collapse_and_reset_break():
    segs = [seg("jungle", 40, 70, region_team="JGL_PR", farming=True),
            seg("jungle", 75, 100, region_team="JGL_PR", farming=True),
            seg("reset", 105, 112),
            seg("base", 113, 125),
            seg("jungle", 130, 160, region_team="JGL_PR", farming=True)]
    d = lexicon.decisions_from_segments(segs)
    toks = [x["token"] for x in d]
    # farmear dos veces seguidas = una decisión; tras el reset vuelve a contar
    assert toks == ["FARMEAR_PROPIA_ROJA", "RESET", "FARMEAR_PROPIA_ROJA"]


def test_all_tokens_have_display():
    for t in lexicon.ALL_TOKENS:
        assert t in lexicon.DISPLAY
