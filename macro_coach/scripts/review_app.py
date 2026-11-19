"""UI local de revisión para el owner: recorre los puntos de decisión de una
partida (tres minimapas, builds, info_pct, decisión real y top-3 del modelo si
hay artefacto) y guarda veredictos en data/feedback/feedback.jsonl.

Uso: .venv\\Scripts\\python.exe scripts\\review_app.py   ->  http://127.0.0.1:8010
"""
import json
from html import escape
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from app.config import CORPUS, ML, FEEDBACK, FEATURES_V, LEXICON_V, ensure_dirs
from app.lexicon import DISPLAY
from app.winprob_review import (CurveStore, IMPORT_DIRECTORY, ReviewArtifactError,
                               ReviewSplitError, read_review_allowlist, preceding_point)

app = FastAPI()


@app.middleware("http")
async def sin_cache(request, call_next):
    # Herramienta en desarrollo: el navegador cacheaba tanto el HTML como el
    # JSON de /api/decisions, y las secciones nuevas (compras) no aparecian
    # aunque el servidor ya las devolviera. Nada de esta app debe cachearse.
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp
_model = None
WINPROB_IMPORT = IMPORT_DIRECTORY


def review_ids():
    try:
        return read_review_allowlist(ML / "split.json")[0]
    except ReviewSplitError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def require_review_game(mid: str):
    # Validate split metadata before touching any per-game corpus file.
    if mid not in review_ids():
        raise HTTPException(status_code=404, detail="Partida no disponible para revisión.")


def review_directory(mid: str):
    directory = CORPUS / mid
    if directory.is_symlink() or not directory.resolve().is_relative_to(CORPUS.resolve()):
        raise HTTPException(status_code=404, detail="Ruta de revisión no permitida.")
    return directory


def review_curve(mid: str):
    try:
        return CurveStore(WINPROB_IMPORT, ML / "split.json").game(mid)
    except (ReviewArtifactError, ReviewSplitError) as error:
        return {"status": "unavailable", "reason": str(error)}


def incompatibilidad(meta: dict, feature_names: list) -> str | None:
    """Motivo por el que el artefacto NO puede proponer, o None si es usable.

    Un modelo de otra versión de features no falla: model_top3 rellena sus
    feature_names con `.get(n, -1)` — sin error y sin aviso — y el top-3 sale
    con buena cara. Pasó con el modelo f2 tras retirar el oro omnisciente en
    f3: 11 features a -1, propuestas de aspecto normal, y cada veredicto del
    owner guardándolas en el feedback. La versión se comprueba como en
    dataset.export() y validate_split(): rechazar, nunca adaptar en silencio.
    """
    from app import dataset
    if (meta.get("features_v") != FEATURES_V
            or meta.get("lexicon_v") != LEXICON_V):
        return (f"modelo features {meta.get('features_v')} / lexicón "
                f"{meta.get('lexicon_v')} != código {FEATURES_V} / {LEXICON_V}"
                " — reentrena: scripts/train_baseline.py")
    sobran = [n for n in feature_names if n not in set(dataset.feature_names())]
    if sobran:
        return (f"el modelo espera features que ya no existen: {sobran[:4]}"
                " — reentrena: scripts/train_baseline.py")
    return None


def model():
    global _model
    f = ML / "model.pkl"
    if _model is None and f.exists():
        with open(f, "rb") as fh:
            _model = pickle.load(fh)
        meta_f = ML / "model.json"
        meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
        _model["smoke"] = bool(meta.get("smoke"))
        _model["incompatible"] = incompatibilidad(meta, _model["feature_names"])
    return _model


def model_top3(decision: dict):
    m = model()
    if m is None:
        return None
    if m.get("incompatible"):
        # la revisión sigue sirviendo (decisiones, oro, compras), pero sin
        # propuestas: mejor ninguna que una calculada sobre huecos a -1
        return {"incompatible": m["incompatible"]}
    from app import dataset
    feats = dataset.feature_vector(decision)
    x = np.array([[feats.get(n, -1) for n in m["feature_names"]]])
    proba = m["clf"].predict_proba(x)[0]
    idx = np.argsort(-proba)[:3]
    return {"smoke": m["smoke"],
            "top": [{"token": m["classes"][i], "p": round(float(proba[i]), 3)}
                    for i in idx]}


@app.get("/", response_class=HTMLResponse)
def index():
    allowed = review_ids()
    rows = []
    for d in sorted(CORPUS.iterdir()) if CORPUS.exists() else []:
        if d.name not in allowed or d.is_symlink():
            continue
        f = d / "decisions.jsonl"
        if f.is_symlink() or not f.exists():
            continue
        seed = {}
        meta_f = d / "meta.json"
        if meta_f.exists() and not meta_f.is_symlink():
            seed = json.loads(meta_f.read_text(encoding="utf-8")).get("seed") or {}
        povs = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                povs.setdefault((r["pid"], r["champ"]), 0)
                povs[(r["pid"], r["champ"])] += 1
        for (pid, champ), n in sorted(povs.items()):
            badge = (f' <b style="color:#fc7">[{escape(str(seed["label"]))}]</b>'
                     if seed.get("pid") == pid else "")
            rows.append(f'<li><a href="/game/{d.name}/{pid}">{d.name} — '
                        f'{escape(str(champ))} (pid {pid})</a>{badge} · {n} decisiones</li>')
    return ("<html><body style='font-family:sans-serif;background:#111;color:#eee'>"
            "<h2>macro_coach — revisión de decisiones</h2><ul>"
            + "".join(rows) + "</ul></body></html>")


@app.get("/img/{mid}/{pass_name}/{fname}")
def img(mid: str, pass_name: str, fname: str):
    require_review_game(mid)
    if pass_name not in {"pass_omni", "pass_blue", "pass_red"} or Path(fname).name != fname or ".." in fname:
        raise HTTPException(status_code=404, detail="Imagen no disponible.")
    directory = review_directory(mid)
    f = directory / pass_name / fname
    if not f.resolve().is_relative_to(directory.resolve()) or not f.exists():
        return JSONResponse({"error": "no"}, status_code=404)
    return FileResponse(str(f))


_ORO = {}


def oro_de(mid: str):
    """`gold.jsonl` de la partida: [(t, [10 x (actual,total)|None])].

    Se lee aparte del snapshot a proposito. El oro exacto del HUD es nuevo y
    todavia no forma parte del esquema del snapshot; meterlo aqui deja al owner
    revisarlo YA sin tocar SNAPSHOT_V ni invalidar las decisiones de las otras
    partidas. Cuando entre en el snapshot, esto sobra.
    """
    if mid not in _ORO:
        f = CORPUS / mid / "gold.jsonl"
        filas = []
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    filas.append((r["t"], r["gold"]))
        _ORO[mid] = sorted(filas)
    return _ORO[mid]


def oro_en(mid: str, t: float):
    """Ultima lectura de oro ANTERIOR a t (causal), o None."""
    filas = oro_de(mid)
    if not filas:
        return None
    prev = [f for f in filas if f[0] <= t]
    if not prev:
        return None
    dt = t - prev[-1][0]
    return {"t": prev[-1][0], "edad_s": round(dt, 1), "gold": prev[-1][1]}


_BUYS = {}


def buys_de(mid: str, pid: int, t0: float, t1: float) -> list:
    """Compras del POV que preparan esta decisión, con su oro exacto al comprar.

    Es lo que pidió el owner: el oro EN el instante de comprar. Sale de
    `buys.jsonl` (app/buys.py). Las compras ocurren en base, en el HUECO entre
    segmentos de decisión (el recall no es una decisión de macro etiquetada),
    así que no se filtra por [t0,t1] —perdería casi todas— sino por la última
    tanda de tienda ANTES de la decisión, hasta 90 s atrás: es la build con la
    que el jugador toma esta decisión.
    """
    if mid not in _BUYS:
        f = CORPUS / mid / "buys.jsonl"
        filas = [json.loads(l) for l in open(f, encoding="utf-8")] \
            if f.exists() else []
        _BUYS[mid] = filas
    prev = [b for b in _BUYS[mid]
            if b["pid"] == pid and b["tipo"] == "compra" and b["t"] <= t0]
    if not prev:
        return []
    ultimo = max(b["t"] for b in prev)
    return [{"t": round(b["t"], 1), "item": b["item"], "hace_s": round(t0 - b["t"], 1),
             "oro": b.get("oro_antes"), "intervalo": b.get("intervalo"),
             "verificado": b.get("verificado", False)}
            for b in prev if ultimo - b["t"] <= 20.0]     # una tanda de tienda


@app.get("/api/decisions/{mid}/{pid}")
def decisions(mid: str, pid: int):
    require_review_game(mid)
    if pid not in range(1, 11):
        raise HTTPException(status_code=404, detail="Perspectiva no disponible.")
    path = review_directory(mid) / "decisions.jsonl"
    if path.is_symlink() or not path.is_file():
        raise HTTPException(status_code=404, detail="Decisiones no disponibles.")
    curve = review_curve(mid)
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["pid"] != pid:
                continue
            snapshot_t = r["snapshot"].get("t")
            pov_team = r["snapshot"].get("me", {}).get("team")
            marker = (preceding_point(curve["points"], snapshot_t * 1000, pov_team)
                      if curve["status"] == "available" and type(snapshot_t) in (int, float) else None)
            if marker is not None:
                marker.update(artifact_sha256=curve["artifact_sha256"], model_digest=curve["model_digest"])
            out.append({
                "t0": r["t0"], "t1": r["t1"], "token": r["token"],
                "snapshot_t": snapshot_t, "pov_team": pov_team, "winprob_marker": marker,
                "display": DISPLAY.get(r["token"], r["token"]),
                "minimap": r["snapshot"]["minimap"],
                "vision": r["snapshot"]["vision"],
                # region_team = verdad a posteriori (para revisar);
                # observed = lo único que vio el POV (lo que come el modelo)
                "players": [dict({k: p.get(k) for k in
                                  ("champ", "role", "team", "level", "items",
                                   "item_gold", "dead", "region_team")},
                                 observed=p.get("observed") or {})
                            for p in r["snapshot"]["players"]],
                "objectives": r["snapshot"]["objectives"],
                "camps": r["snapshot"]["camps"],
                "model_top3": model_top3(r),
                "oro": oro_en(mid, r["t0"]),
                "compras": buys_de(mid, pid, r["t0"], r["t1"]),
            })
    return out


@app.get("/api/winprob/{mid}")
def winprob(mid: str):
    require_review_game(mid)
    return review_curve(mid)


@app.get("/winprob/model", response_class=HTMLResponse)
def winprob_model():
    try:
        bundle = CurveStore(WINPROB_IMPORT, ML / "split.json").load()
    except (ReviewArtifactError, ReviewSplitError) as error:
        return HTMLResponse("<h2>Curva no disponible</h2><p>" + escape(str(error)) + "</p>", status_code=503)
    meta = bundle["meta"]
    validation = meta["model_validation"]
    declared = {"validated_original_snapshot_development": "Evaluación de desarrollo; sin confirmación independiente."}.get(
        validation["state"], validation["state"])
    return HTMLResponse("<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:850px;margin:32px;background:#111;color:#eee'>"
        "<a href='/' style='color:#7ac'>Volver a la revisión</a><h2>Validación de la curva de victoria</h2>"
        "<p>" + escape(declared) + "</p><p>" + escape(validation["description"]) + "</p>"
        "<p>Estimación retrospectiva OMNI del estado de la partida. Un descenso no identifica una decisión errónea, "
        "su causa ni una alternativa mejor. No se ha establecido beneficio de entrenamiento para el jugador.</p>"
        "<details><summary>Procedencia del artefacto importado</summary><p>Modelo: <code>" + escape(meta["model_digest"]) + "</code></p>"
        "<p>Informe de origen: <code>" + escape(validation["report_path"]) + "</code></p>"
        "<p>SHA256 del informe: <code>" + escape(validation["report_sha256"]) + "</code></p>"
        "<p>Esta página usa exclusivamente los metadatos copiados; no abre el informe de origen.</p></details></body></html>")


@app.post("/api/feedback")
async def feedback(payload: dict):
    require_review_game(payload.get("match_id", ""))
    ensure_dirs()
    payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(FEEDBACK / "feedback.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"ok": True}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>revisión</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
.maps{display:flex;gap:8px}.maps figure{margin:0}.maps img{width:300px;image-rendering:pixelated}
figcaption{font-size:12px;color:#999;text-align:center}
.tok{font-size:20px;margin:10px 0}.tok b{color:#7fc}
.model li{color:#fc7}.bar{background:#333;height:10px;width:200px;display:inline-block}
.bar div{background:#7fc;height:10px}
button{margin:4px;padding:6px 14px;background:#222;color:#eee;border:1px solid #555;cursor:pointer}
button.on{background:#264}
table{font-size:12px;border-collapse:collapse}td,th{padding:2px 6px;border-bottom:1px solid #333}
.dead{color:#f66}
.wp{max-width:980px;margin:20px 0;padding:18px;border:1px solid #344448;border-radius:12px;background:#131b1d}
.wp h3{margin:0 0 6px}.wp p{font-size:13px;line-height:1.5;margin:7px 0;color:#abbabd}
.wp-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.wp svg{display:block;width:100%;max-height:290px}.wp a{color:#8dcced}
.wp-current{color:#d8f4eb;font-size:14px;margin:5px 0}.wp-falls{display:flex;flex-wrap:wrap;gap:5px}
.wp-falls button{text-align:left;font-size:12px;border-radius:6px;border-color:#425255}
.wp-note{font-size:12px;color:#99abad}.wp-warn{color:#edc98c}
</style></head><body>
<a href="/" style="color:#7ac">&larr; partidas</a>
<h3 id="head"></h3>
<div class="tok" id="tok"></div>
<div class="maps" id="maps"></div>
<div id="vision"></div>
<div id="compras"></div>
<section class="wp" aria-labelledby="wp-title">
<div class="wp-head"><div><h3 id="wp-title">Probabilidad estimada de victoria</h3>
<p>Revisión retrospectiva OMNI. La curva describe el estado de la partida; no atribuye el cambio a esta decisión.</p></div>
<div><button id="wp-pov" class="on" onclick="setWpMode('pov')">Equipo del POV</button>
<button id="wp-blue" onclick="setWpMode('blue')">Azul</button></div></div>
<div id="wp-status" class="wp-note">Cargando artefacto local…</div>
<div id="wp-chart"></div><div id="wp-current" class="wp-current"></div>
<div id="wp-falls" class="wp-falls"></div>
<p id="wp-window-note"></p><p><a href="/winprob/model" target="_blank" rel="noopener">Estado y validación del modelo</a></p>
</section>
<div class="model" id="model"></div>
<div>
<button onclick="verdict('agree')">de acuerdo</button>
<button onclick="verdict('disagree')">en desacuerdo</button>
<button onclick="verdict('unclear')">dudoso</button>
<input id="note" placeholder="nota" style="width:300px;background:#222;color:#eee;border:1px solid #555">
</div>
<p><button onclick="step(-1)">&larr; anterior</button>
<span id="pos"></span>
<button onclick="step(1)">siguiente &rarr;</button></p>
<table id="players"></table>
<script>
const [mid, pid] = [location.pathname.split('/')[2], location.pathname.split('/')[3]];
let D = [], i = 0, WP = null, wpMode = 'pov';
function mmss(t){t=Math.round(t);return String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0')}
function setWpMode(mode){wpMode=mode;document.getElementById('wp-pov').classList.toggle('on',mode==='pov');document.getElementById('wp-blue').classList.toggle('on',mode==='blue');if(D.length)renderWp(D[i]);}
function renderWp(d){
  const status=document.getElementById('wp-status'), chart=document.getElementById('wp-chart');
  const current=document.getElementById('wp-current'), cards=document.getElementById('wp-falls');
  const note=document.getElementById('wp-window-note');
  chart.innerHTML='';current.textContent='';cards.innerHTML='';note.textContent='';
  if(!WP || WP.status!=='available'){status.textContent=WP?.reason||'Curva no disponible para esta partida.';return;}
  const team=wpMode==='blue'?100:d.pov_team;
  if(team!==100 && team!==200){status.textContent='No se conoce el equipo de esta perspectiva.';return;}
  const exposure={fit:'partida utilizada para ajustar el modelo',calibration:'partida utilizada para calibrar',development:'partida de desarrollo ya evaluada',unseen:'partida ajena al ajuste y la calibración'};
  status.textContent=(team===100?'Equipo azul':'Equipo rojo')+' · '+exposure[WP.exposure_role]+'. Muestras por minuto; no es una estimación en vivo.';
  const points=WP.points, width=900,height=235,left=43,right=15,top=15,bottom=31;
  const first=points[0].timestamp_ms,last=Math.max(first+60000,points[points.length-1].timestamp_ms);
  const X=t=>left+(t-first)/(last-first)*(width-left-right), Y=p=>top+(1-p)*(height-top-bottom);
  const P=p=>team===100?p:1-p;
  let axes='';
  for(const probability of [0,.25,.5,.75,1]) axes+='<line x1="'+left+'" y1="'+Y(probability)+'" x2="'+(width-right)+'" y2="'+Y(probability)+'" stroke="'+(probability===.5?'#476068':'#2b3a3e')+'"/><text x="'+(left-7)+'" y="'+(Y(probability)+4)+'" fill="#9caaae" text-anchor="end" font-size="11">'+Math.round(probability*100)+'%</text>';
  for(let tick=0;tick<=4;tick++){const time=first+(last-first)*tick/4;axes+='<text x="'+X(time)+'" y="'+(height-8)+'" fill="#9caaae" text-anchor="middle" font-size="11">'+mmss(time/1000)+'</text>';}
  let path='';points.forEach((point,index)=>{const x=X(point.timestamp_ms),y=Y(P(point.p_blue));
    path+=(index===0||point.timestamp_ms-points[index-1].timestamp_ms>90000?' M '+x+' '+y:' H '+x+' V '+y);});
  let marker='';const observed=d.winprob_marker?.artifact_sha256===WP.artifact_sha256?d.winprob_marker:null;
  if(Number.isFinite(d.snapshot_t)&&d.snapshot_t*1000>=first&&d.snapshot_t*1000<=last){const xx=X(d.snapshot_t*1000);marker+='<line x1="'+xx+'" y1="'+top+'" x2="'+xx+'" y2="'+(height-bottom)+'" stroke="#f0cf93" stroke-dasharray="4 4"/>';}
  if(observed){marker+='<circle cx="'+X(observed.timestamp_ms)+'" cy="'+Y(P(observed.p_blue))+'" r="5" fill="#f0cf93" stroke="#172427" stroke-width="2"/>';
    current.textContent='Último punto anterior: '+(100*P(observed.p_blue)).toFixed(1)+'% a '+mmss(observed.timestamp_ms/1000)+' · antigüedad '+(observed.age_ms/1000).toFixed(1)+' s'+(observed.stale?' · punto antiguo; no hay una estimación actual.':'');
    current.classList.toggle('wp-warn',observed.stale);
  }else{current.textContent='Sin muestra anterior al corte de esta decisión.';current.classList.remove('wp-warn');}
  chart.innerHTML='<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Curva retrospectiva de probabilidad estimada de victoria, equipo '+(team===100?'azul':'rojo')+'"><title>Estimación OMNI por minuto; línea discontinua: corte de la decisión; punto: última muestra anterior</title>'+axes+'<path d="'+path+'" fill="none" stroke="#77dbb4" stroke-width="2.5"/>'+marker+'</svg>';
  const falls=WP.falls_by_team[String(team)]||[];
  for(const window of falls){const button=document.createElement('button');
    button.textContent='Descenso estimado '+window.delta_team_pp.toFixed(1)+' pp · '+mmss(window.start_ms/1000)+'–'+mmss(window.end_ms/1000);
    let target=-1;D.forEach((decision,index)=>{if(decision.t0*1000<=window.start_ms&&(target<0||decision.t0>D[target].t0))target=index;});
    button.disabled=target<0;button.title=target<0?'No hay decisión anterior a esta ventana.':'Abrir la última decisión anterior al inicio de la ventana.';
    button.onclick=()=>{i=target;render();};cards.appendChild(button);}
  note.textContent=falls.length?'Hasta tres ventanas observadas de unos 3 minutos, sin solapamiento. Navegan al contexto previo; no señalan errores ni efectos de una acción.':'No hay descensos estimados en ventanas observadas de tres minutos para este equipo.';
}
function render(){
  if(!D.length){document.getElementById('head').textContent='Sin decisiones disponibles para esta perspectiva.';return;}
  const d = D[i];
  renderWp(d);
  document.getElementById('head').textContent = mid+' · pid '+pid+' · '+mmss(d.t0)+'-'+mmss(d.t1);
  document.getElementById('pos').textContent = (i+1)+' / '+D.length;
  document.getElementById('tok').innerHTML = 'decisión del challenger: <b>'+d.display+'</b> ('+d.token+')';
  const maps = document.getElementById('maps'); maps.innerHTML='';
  for (const [name,label] of [['pass_omni','sin niebla'],['pass_blue','niebla azul'],['pass_red','niebla roja']]){
    const ref = d.minimap[name];
    const fig = document.createElement('figure');
    fig.innerHTML = ref ? '<img src="/img/'+mid+'/'+ref.file+'"><figcaption>'+label+' (dt '+ref.dt+'s)</figcaption>'
                        : '<div style="width:300px;height:300px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;color:#555">sin pasada</div><figcaption>'+label+'</figcaption>';
    maps.appendChild(fig);
  }
  const v = d.vision;
  const fmtE = e => e.champ+': '+(e.known ? (e.conf+(e.dead?' (muerto)':'')) : '<i>desconocido</i>');
  document.getElementById('vision').innerHTML = !v.available
    ? 'visibilidad no disponible (sin pasadas con niebla)'
    : (v.info_pct === null || v.info_pct === undefined
        ? '<b style="color:#f66">visibilidad NO utilizable</b>'
          + (v.reason ? ' ('+v.reason+')' : '')
          + ' — conocidos '+(v.n_known??0)+'/'+(v.n_enemies??5)+'<br>'
          + (v.enemies||[]).map(fmtE).join(' · ')
        : 'info de los conocidos ('+v.n_known+'/'+v.n_enemies+'): <b>'+v.info_pct+'%</b>'
          + ' <span class="bar"><div style="width:'+2*v.info_pct+'px"></div></span> '
          + (v.enemies||[]).map(fmtE).join(' · '));
  // Compras del POV en este segmento, con el oro EXACTO al comprar (o el
  // intervalo si el ingreso no se pudo localizar). Es el dato que pidio el
  // owner: la cartera EN el instante de la compra, no la de hace unos segundos.
  const cmp = d.compras || [];
  document.getElementById('compras').innerHTML = cmp.length
    ? '<b>última tienda del POV antes de decidir</b> (hace '+cmp[0].hace_s+'s): ' + cmp.map(b=>{
        const oro = b.oro!=null ? '<b>'+b.oro+' oro</b>'
          : (b.intervalo ? '<span style="color:#fc7">'+b.intervalo[0]+'–'+b.intervalo[1]+' oro</span>'
                         : '<span style="color:#f66">?</span>');
        return 'item '+b.item+' con '+oro
          + (b.verificado ? '' : ' <span style="color:#888">(acotado)</span>');
      }).join(' · ')
    : '<span style="color:#888">sin compras previas del POV</span>';
  document.getElementById('model').innerHTML = d.model_top3
    ? (d.model_top3.incompatible
       ? '<b style="color:#f66">modelo incompatible, sin propuestas: '+d.model_top3.incompatible+'</b>'
       : (d.model_top3.smoke ? '<b style="color:#f66">modelo SMOKE (fontanería, ignorar las propuestas)</b>' : 'modelo propone:')
         + '<ol>'+d.model_top3.top.map(m=>'<li>'+m.token+' ('+(100*m.p).toFixed(0)+'%)</li>').join('')+'</ol>')
    : '<i>sin modelo entrenado</i>';
  const tbl = document.getElementById('players');
  const oro = d.oro;
  // La lectura de oro es la ULTIMA anterior a la decisión (causal): se dice de
  // cuándo es, para que el owner sepa si mira un dato de hace 2s o de hace 8.
  const cabOro = oro
    ? '<th>oro HUD<br><small>actual(total) · hace '+oro.edad_s+'s</small></th>'
    : '<th>oro HUD<br><small>sin lectura</small></th>';
  tbl.innerHTML = '<tr><th>campeón</th><th>rol</th><th>equipo</th><th>nivel</th>'
      + '<th>región real</th><th>lo que sabía el POV</th>' + cabOro
      + '<th>oro items</th><th>items</th></tr>'
    + d.players.map((p,idx)=>{
        const o = p.observed||{};
        const saw = !o.known ? '<span style="color:#f66">desconocido</span>'
          : (o.source==='ultimo_avistamiento'
              ? (o.region_team||'?')+' <span style="color:#fc7">(visto hace '+o.age_s+'s)</span>'
              : (o.source==='muerto' ? 'muerto' : (o.region_team||'?')));
        // El oro del HUD viene de gold.jsonl, por indice de participante.
        // Sin lectura (partida capturada sin oro, o celda ilegible) se dice,
        // no se rellena con un numero.
        const g = oro && oro.gold ? oro.gold[idx] : null;
        const celdaOro = g
          ? g[0]+'<span style="color:#888">('+g[1]+')</span>'
          : '<span style="color:#f66">—</span>';
        return '<tr class="'+(p.dead?'dead':'')+'"><td>'+p.champ+'</td><td>'+p.role+'</td><td>'
          +(p.team==100?'azul':'rojo')+'</td><td>'+(p.level??'?')+'</td><td>'+(p.region_team??'?')
          +'</td><td>'+saw+'</td><td>'+celdaOro+'</td><td>'+p.item_gold+'</td><td>'
          +p.items.join(' ')+'</td></tr>';
      }).join('');
}
function step(k){if(!D.length)return;i = Math.min(Math.max(i+k,0), D.length-1); render(); }
function verdict(v){
  if(!D.length)return;
  const d = D[i];
  fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({match_id:mid,pid:+pid,t0:d.t0,token:d.token,verdict:v,
      model_top3:d.model_top3,note:document.getElementById('note').value})})
    .then(()=>{document.getElementById('note').value='';step(1)});
}
Promise.all([
  fetch('/api/decisions/'+mid+'/'+pid).then(async r=>{if(!r.ok)throw new Error((await r.json()).detail||'Revisión no disponible');return r.json();}),
  fetch('/api/winprob/'+mid).then(r=>r.json()).catch(()=>({status:'unavailable',reason:'No se pudo cargar la curva local.'}))
]).then(([decisions,curve])=>{D=decisions;WP=curve;render();}).catch(error=>{document.getElementById('head').textContent=error.message;document.getElementById('wp-status').textContent='Revisión no disponible.';});
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)});
</script></body></html>"""


@app.get("/game/{mid}/{pid}", response_class=HTMLResponse)
def game(mid: str, pid: int):
    require_review_game(mid)
    if pid not in range(1, 11):
        raise HTTPException(status_code=404, detail="Perspectiva no disponible.")
    # sin caché: la página es una herramienta en desarrollo; el navegador se
    # quedaba con el JS viejo y no aparecían secciones nuevas (las compras)
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    ensure_dirs()
    uvicorn.run(app, host="127.0.0.1", port=8010)
