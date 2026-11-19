"""Rutas y constantes del proyecto. Todo umbral auditado en fase 1 vive aquí
con su valor original; cambiarlo invalida la regresión contra la partida
prototipo."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
CACHE = DATA / "cache"
ML = DATA / "ml"
REPORTS = DATA / "reports"
FEEDBACK = DATA / "feedback"
QUEUE_FILE = DATA / "queue.jsonl"


def load_env():
    """Lee .env plano (KEY=VALUE) sin depender de dotenv en import-time."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()

RIOT_API_KEY = os.environ.get("RIOT_API_KEY", "")
PLATFORM = "euw1"            # replays solo descargables en la región del cliente
REGIONAL = "europe"
QUEUE_RANKED_SOLO = 420

# --- mapa y regiones (fase 1, auditado) ---
MAP_MAX = 14870.0
LBN = {1: "BASE", 2: "BASE_ENEMIGA", 3: "TOP", 4: "MID", 5: "BOT",
       6: "RIO_TOP", 7: "RIO_BOT", 8: "JGL_PA", 9: "JGL_PR",
       10: "JGL_EA", 11: "JGL_ER", 12: "PIT_BARON", 13: "PIT_DRAGON"}
REGION_MASK_FILE = DATA / "region_labels.npy"

# campamentos (coordenadas normalizadas, lado azul; el rojo es el espejo)
CAMPS_BLUE = {"gromp": (0.142, 0.568), "azul": (0.256, 0.531),
              "lobos": (0.254, 0.433), "raptors": (0.467, 0.342),
              "rojo": (0.525, 0.272), "krugs": (0.563, 0.181)}

# --- segmentación (umbrales auditados de macro_v5) ---
# Segundo filtro de score sobre las detecciones ya aceptadas por el detector.
# Fase 1 usaba 0.895 con retratos del HUD; con los de ddragon las puntuaciones
# dependen del campeon (Shaco: mediana 0.863, TODAS sus 511 detecciones caian
# bajo 0.895 y desaparecia del dataset sin error alguno). Se iguala al umbral
# del detector para no tener dos puertas incoherentes.
TRAJ_MIN_SCORE = 0.82
TRAJ_MIN_COVERAGE = 0.50     # QA: fraccion de frames con deteccion por campeon
GANK_RADIUS = 0.15           # rival "cerca" para gank/pelea
SEG_MIN_S = 8                # segmento mínimo sin ancla
SEG_JOIN_S = 10              # hueco máximo dentro de un segmento
SEG_MERGE_S = 14             # hueco máximo para fundir segmentos iguales
FARM_JCS_MIN = 3             # subida de CS de jungla que cuenta como farmear
GANK_ENEMY_HALF = 0.97       # cx+cy >= esto = mitad enemiga
CAMP_WINDOW = (-5.0, 15.0)   # ventana rombo-muerto alrededor del segmento

# --- visibilidad / métrica de información (fase 2, v1) ---
CONF_HORIZON_S = 45.0        # decaimiento lineal hasta 0 en 45s sin ver
CONF_VISIBLE_GRACE_S = 4.0   # visto hace <= esto cuenta como visible ahora
VIS_VERIFY_SCORE = 0.84      # umbral de verificación local en pasadas con niebla
VIS_VERIFY_WIN_PX = 24       # media ventana de búsqueda alrededor de la posición real

# --- detector ---
# Fase 1: umbral 0.87 y tamanos fijos (38,44,50) con retratos del HUD sobre un
# minimapa de 508px. Fase 2 usa retratos de ddragon (no hay HUD recortado para
# 170 campeones) y minimapas de tamano variable: los tamanos se derivan del
# minimapa calibrado y el umbral baja a 0.82. Medido sobre EUW1_7986655074
# t=420: fijos+0.87 -> 8/10; proporcionales+0.82 -> 10/10.
DET_CLS_MIN = 0.82
DET_NMS_DIST = 16
DET_MIN_AREA = 90
DET_SIZE_FRACS = (0.060, 0.069, 0.078, 0.087, 0.095, 0.104, 0.113, 0.121)
DET_SIZES = (38, 44, 50)          # fallback si no se conoce el tamano del mapa
DET_ZOOM = 1.2

# --- captura ---
CAPTURE_MON = {"left": 0, "top": 0, "width": 1920, "height": 1080}
CAPTURE_CROP = (540, 1060, 1380, 1905)   # y0,y1,x0,x1 del recorte del minimapa
MM_GEOM_DEFAULT = (9, 5, 508)            # x0,y0,size dentro del recorte
PLAYERLIST_EVERY_S = 8.0
REPLAY_API = "https://127.0.0.1:2999"
# 32x, medido en pasadas COMPLETAS reales sobre EUW1_7986655074:
#   8x  -> 3.8 min/pasada, 540 frames, densidad 2.90s (p90 3.10s), 9.5/10 det
#   16x -> 2.1 min,        522 frames, densidad 2.90s (p90 3.40s), 9.9/10
#   32x -> 1.3 min,        520 frames, densidad 2.80s (p90 3.80s), 10.0/10
#   64x -> 0.9 min,        341 frames, densidad 4.50s (p90 6.90s), 9.7/10  <-- NO
# 64x adelgaza la muestra por encima de 3.5s, que es donde la sensibilidad de
# la visibilidad se desploma (81.7% -> 50%, docs/history.md). 32x mantiene la
# densidad de 8x en un tercio del tiempo.
# Requisitos para que 32x aguante (si no, bajar a 16x u 8x):
#   - escritura de PNG en hilo aparte (si no, el disco bloquea el bucle),
#   - conexion HTTPS persistente al reloj de la replay (si no, el suelo del
#     bucle sube a ~114ms y la densidad cae a 3.6s),
#   - NO procesar mientras se captura (con los 16 nucleos ocupados salia ~97%
#     de frames duplicados).
REPLAY_SPEED = 32
# Densidad objetivo en segundos DE PARTIDA por muestra; la espera del bucle se
# deriva de ella y de la velocidad.
SAMPLE_EVERY_GAME_S = 2.4
CAPTURE_LOOP_OVERHEAD_S = 0.035

# --- objetivos: constantes de spawn (parche 16.x — verificar en cambio de parche) ---
DRAGON_FIRST_S = 300.0
DRAGON_RESPAWN_S = 300.0
GRUBS_SPAWN_S = 360.0
HERALD_SPAWN_S = 840.0       # ventana Larvas->Heraldo usada por el lexicón
BARON_SPAWN_S = 1200.0       # ventana Heraldo->Barón usada por el lexicón

# --- dataset / modelo ---
LEXICON_V = "lex1"
# f2: las features de enemigos pasan a ser SOLO lo observable por el equipo del
# POV (antes se enviaba la región real de los cinco enemigos, sacada de la
# pasada sin niebla, también cuando estaban ocultos).
# f3: fuera el oro de la capa de verdad (totalGold de los diez y gold_diff_pf:
# ningún jugador ve el oro total, solo el espectador). La diferencia de oro es
# ahora la estimación OBSERVABLE: item_gold_diff (builds visibles en el Tab)
# + cartera propia (my_wallet_pf). Campamentos: sin cambio (decisión del owner
# 2026-09-20: se quedan aunque vengan de la pasada omni).
FEATURES_V = "f3"
# s2: el snapshot separa `truth` (retrospectiva, para etiquetar y revisar) de
# `observed` (lo que el equipo podía saber al decidir) y usa lookup causal.
SNAPSHOT_V = "s2"
MIN_GAMES_REPORT = 30
SPLIT_FRACS = (0.70, 0.15, 0.15)

# --- calidad de la visibilidad ---
# Sensibilidad mínima (detecciones del propio equipo, donde el campeón SIEMPRE
# está dibujado) para que su no-detección signifique algo. Por debajo, la
# visibilidad de ese campeón es DESCONOCIDA, no "no visible": con s=0.3 la
# ausencia es casi ruido.
# Medido con scripts/vis_quality.py sobre el corpus: mediana 0.73, p25 0.65,
# p05 0.28 — hay una cola corta de campeones que el detector casi no ve. 0.50
# cae en una meseta (0.4 y 0.5 dejan el 90% utilizable; 0.6 -> 87%, 0.7 -> 67%),
# así que recorta esa cola sin tirar datos buenos. Recalibrar con el feedback
# del owner.
VIS_MIN_SENSITIVITY = 0.50
# Enemigos con visibilidad utilizable necesarios para publicar un info_pct.
VIS_MIN_USABLE_ENEMIES = 3

SNAPSHOT_EPS_S = 1.0         # el snapshot se toma en t0 - eps
MINIMAP_REF_MAX_DT = 5.0     # frame más cercano admisible para el snapshot


# Partidas que están en data/corpus pero NO forman parte del corpus de
# entrenamiento. Se conservan porque sirven de otra cosa.
EXCLUDED_MATCHES = {
    # Prototipo de fase 1: partida de soloq del propio owner (Evelynn), no de
    # un jungla challenger; sin pasadas de niebla (vision no disponible) y
    # capturada con el formato antiguo. Se queda SOLO como fixture de
    # regresión de tests/test_regression_prototype.py.
    "EUW1_7983109023",
}


def is_excluded(match_id: str) -> bool:
    return match_id in EXCLUDED_MATCHES


def game_dir(match_id: str) -> Path:
    return CORPUS / match_id


def ensure_dirs():
    for d in (DATA, CORPUS, CACHE, ML, REPORTS, FEEDBACK):
        d.mkdir(parents=True, exist_ok=True)
