# macro_coach

Asistente de macro-coaching post-partida para League of Legends. Parsea una
partida en acciones `(lugar, acción, receptor)` a partir de la replay y de
Match-V5, para poder revisarla después y —más adelante— entrenar sobre ello.

Proyecto hermano de `../build_tracker` (el asesor de tienda). Comparten dominio
y clave de Riot, pero **no comparten código ni datos**.

Estado: **prototipo validado sobre una partida** (`EUW1_7983109023`, Evelynn
jungla, parche 16.18), auditado por el owner corrigiendo ejemplos concretos
hasta que la salida coincidió con lo que pasó. No hay modelo todavía: primero
la extracción, luego el aprendizaje.

## Arquitectura: qué aporta cada fuente

El reparto no es negociable — costó una sesión entera descubrirlo:

| Fuente | Qué aporta | Por qué |
| --- | --- | --- |
| **Match-V5** (`/lol/match/v5`) | Kills, objetivos y edificios **con posición exacta**; `participantFrames` cada 60s con posición y CS exactos de los 10 | Autoritativa y reproducible. Cada acción se sitúa **donde ocurrió**, por posición |
| **Minimapa de la replay** | Trayectoria densa (~3s) de los 10; estado de cada campamento por el rombo naranja | Es lo único que la API no observa entre frames de 60s |
| ~~Live Client `eventdata`~~ | **No usar** | Acumula duplicados entre pasadas (llegó a marcar 9 kills donde había 6) y pierde objetivos con los seeks. Solo sería fiable en una pasada continua sobre una replay recién abierta |

Coste de extracción: **una sola pasada a 8x por partida, ~3,5 min**, que alimenta
a la vez trayectorias, rombos de campamentos y `playerlist`.

## Taxonomía v1

**Lugares** (13 regiones, derivadas de los píxeles reales del minimapa):
base propia y enemiga (cuarto de círculo completo, r≈0.37 medido en el color del
suelo), TOP / MID / BOT (polilíneas con su codo, que se ensanchan hacia su tramo
de río), río superior e inferior (por el color azul real), foso de Barón y foso
de Dragón, y los cuatro cuadrantes de jungla (propia/enemiga × azul/rojo).

**Acciones**: farmear jungla (por cuadrante), invadir, gank, pelea en un carril
del lado propio, río/foso, backear, estar en base, y objetivo (de cualquiera de
los dos equipos). Las acciones se encadenan cuando corresponde —
`GANK TOP → INVADE jungla enemiga · cuadrante rojo`.

**Receptor**: sale de la posición del evento y de quién lo recibe
(`Cho'Gath (TOP)`, `Torre BOT`, `Dragón quimtec`).

### Reglas que distinguen lo real de lo aparente

- **Gank** = pisar un carril con rival detectado cerca (<0.15) o con un kill /
  muerte anclado. En la mitad propia del mapa no es gank, es `pelea en X (lado propio)`.
- **Farmear ≠ pasar por la jungla**: farmea solo si muere un rombo de ese
  cuadrante durante el segmento o sube el CS de jungla; si no, `cruza`.
- **Un campamento es jungla por definición**: gromp y krugs rozan sus carriles en
  el mapa real, así que `build_regions.py` recorta un disco (r=0.042) en cada uno
  y lo devuelve a su cuadrante. El script **valida los doce** antes de guardar.
- Huecos en la timeline = muerta o sin detección, nunca se rellenan.

## Los scripts

| Script | Qué hace |
| --- | --- |
| `scripts/capture_pass.py` | Pone el juego en primer plano, reproduce la replay entera a 8x y guarda el recorte del minimapa + `playerlist`, sincronizado con el reloj exacto de `/replay/playback` |
| `scripts/analyze_images.py` | Detecta a los 10 campeones en cada frame: máscara del anillo de equipo → componentes conexas → clasificación local contra los retratos |
| `scripts/build_camp_dots.py` | Serie de presencia del rombo naranja por campamento |
| `scripts/build_regions.py` | Construye la máscara de regiones y **valida** que los doce campamentos caigan en jungla |
| `scripts/macro_v5.py` | Compone la timeline final cruzando Match-V5, trayectorias y rombos |

## Antes de capturar

1. `EnableReplayApi=1` en la sección `[General]` de
   `C:\Riot Games\League of Legends\Config\game.cfg`, **antes** de abrir la replay.
2. `RIOT_API_KEY` válida (las de desarrollo caducan cada 24h).
3. No tocar el ratón ni el teclado durante la pasada: si el juego pierde el foco,
   los frames capturan otra ventana. `capture_pass.py` valida cada frame y aborta
   tras 15 inválidos seguidos, pero más vale no provocarlo.
4. Si cambias el tamaño del minimapa a mitad de sesión, la geometría del recorte
   cambia: hay que detectarla por frame, no asumirla.

## Fase 2 — el consejo de junglas (en marcha)

El prototipo escala a corpus: partidas de junglas challenger (EUW, parche en
curso), **tres pasadas por partida** (sin niebla + niebla de cada equipo),
builds de los 10 reconstruidas de Match-V5, métrica de información 0-100 sobre
los 5 enemigos, lexicón cerrado de decisiones (`LEXICON.md`) y un modelo básico
de imitación con revisión del owner. Wards: fuera de alcance por decisión del
owner — la visión entra solo como el efecto neto que muestran u ocultan las
pasadas con niebla.

- **Diseño y plan de recogida**: `docs/design.md`
- **Contrato de decisiones**: `LEXICON.md` (v1, 25 tokens)
- **Contrato para asistentes**: `CLAUDE.md` (comandos, prohibiciones, gotchas)
- **Estado**: `.venv\Scripts\python.exe scripts\status.py`
- Los scripts del prototipo viven en `docs/archive/fase1_scripts/`
  (superseded); sus sucesores son los módulos de `app/`.

La partida auditada de fase 1 (`EUW1_7983109023`) es ahora la regresión del
pipeline (`tests/test_regression_prototype.py`) y la primera del corpus:
97 puntos de decisión (Evelynn 38 + Vi 59), sin visibilidad (aquella sesión no
tuvo pasadas con niebla).

Pendiente heredado de fase 1: los roams de mid/support como POV adicionales, y
auditar la regla nueva del centinela en las primeras partidas de fase 2.
