# Fase 2 — el consejo de junglas challenger

## Objetivo

Reunir la perspectiva de suficientes junglas de challenger (sus decisiones
reveladas, con el estado completo en que las tomaron) para entrenar un modelo
básico que, dado un estado, proponga la decisión que tomaría un challenger.
El owner (Javier) audita las propuestas y da feedback; ese feedback calibra
lexicón, métrica de confianza y modelo.

No optimizamos resultados ni afirmamos que una decisión sea la mejor: es
**imitación conductual**, igual que build_tracker con la tienda.

## Fuentes (heredado del prototipo, no negociable)

| Fuente | Aporta | Nota |
| --- | --- | --- |
| Match-V5 | eventos con posición exacta; frames 60s (posición, oro, CS, nivel); eventos de ítems | autoritativa y reproducible |
| Replay, pasada sin niebla (`pass_omni`) | trayectoria densa (~3–4s) de los 10; rombos de campamentos; playerlist cada 8s (ahora también con ítems) | lo que la API no ve |
| Replay, pasadas con niebla (`pass_blue`, `pass_red`) | **la verdad de visibilidad de cada equipo**: qué campeones enemigos aparecen en su minimapa y cuándo | nuevo en fase 2 |
| Live Client `eventdata` | no usar | duplica entre pasadas (auditado en fase 1) |

**Wards: fuera de alcance por decisión del owner.** `WARD_PLACED` llega sin
coordenadas y los iconos del minimapa no son fiables. No se modelan wards en
ningún punto: la visión entra al sistema solo como su efecto neto — lo que las
pasadas con niebla muestran u ocultan.

## Las tres perspectivas

Tres pasadas de captura por partida, mismas herramientas que la fase 1:

1. `pass_omni` — niebla desactivada. Verdad: trayectorias de los 10, rombos.
2. `pass_blue` — niebla del equipo azul (ORDER). El minimapa muestra solo lo
   que azul ve.
3. `pass_red` — niebla del equipo rojo (CHAOS).

La niebla se fija vía Replay API (`POST /replay/render`, propiedad `fogOfWar`,
con `selectionName` a un campeón del equipo observador). **Validación empírica
obligatoria al inicio de cada pasada**: con niebla del equipo T, los píxeles
del color del equipo rival en el minimapa deben ser ~0 en los primeros
segundos (los enemigos en su fuente no son visibles). Si falla, la pasada
aborta e indica el ajuste manual (selector de niebla del HUD de la replay).
El resultado de la validación queda en `meta.json` de la pasada.

De las pasadas con niebla **no** se re-derivan trayectorias (están ocluidas por
definición). Se deriva una serie de **visibilidad** por enemigo: en cada frame
sabemos dónde está realmente (trayectoria omni) y verificamos localmente si su
icono aparece en esa posición del minimapa con niebla. Verificación local en
ventana pequeña = problema mucho más fácil que la detección global.

## Métrica de información (`info_pct`, 0–100)

Por enemigo, desde la perspectiva del equipo del jungla, en el instante t:

- muerto → **100** (posición conocida: fuente; el respawn es información pública).
- visible ahora (avistamiento hace ≤ 4s, el hueco entre frames) → **100**.
- si no: decaimiento lineal desde el último avistamiento confirmado:
  `max(0, 100 * (1 - dt/45))` con horizonte `H = 45s` (`app/config.py`).
- el inicio de partida siembra un avistamiento en la fuente para los 10
  (el spawn es conocimiento público): sin casos especiales antes del primer
  avistamiento real.

Agregado del equipo = **media de los enemigos CONOCIDOS** (ver abajo). Se
almacenan además los crudos por enemigo (`last_seen_t`, `visible`, `dead`,
`sensitivity`) para que el modelo no dependa de la fórmula.

Sin pasadas de niebla (p. ej. la partida rescatada de fase 1):
`vision.available = false` e `info_pct = null`. **Nunca se aproxima la
visibilidad desde datos omniscientes** — sería inventar una etiqueta.

### Qué significa `conf`, y el desconocido explícito

`conf` es la **frescura del último avistamiento confirmado**, no la
probabilidad de que el enemigo esté oculto. El detector tiene ~20% de falsos
negativos, así que una no-detección aislada no prueba nada. De ahí tres reglas:

1. la **sensibilidad por campeón** se mide en la pasada de su propio equipo
   (donde siempre está dibujado). Si baja de `VIS_MIN_SENSITIVITY`, su
   visibilidad es **desconocida** (`known=false`, `conf=null`), no "no visible";
2. `info_pct` sólo se publica con al menos `VIS_MIN_USABLE_ENEMIES` conocidos;
   si no, es `null`;
3. una pasada marcada `suspect_no_fog` (o con la misma niebla que su pareja)
   **invalida** la visibilidad de esa partida para entrenar.

`visible_frac_corrected` (el cociente pareado) es una **frecuencia agregada de
auditoría**: nunca se convierte en visibilidad instante a instante.

## Dos capas en el snapshot: verdad observada vs. verdad a posteriori

Lo que pasó y lo que el jugador podía saber no son lo mismo, y mezclarlos
arruina el dataset (pasó: 312 de 315 enemigos ocultos llegaban al modelo con su
región real).

- `players[*]` — **verdad a posteriori**, de la pasada sin niebla. Sirve para
  etiquetar el segmento y para que el owner revise.
- `players[*].observed` — **lo observable por el equipo del POV** en ese
  instante: `known`, `source` (`own_team` / `visible` / `ultimo_avistamiento` /
  `muerto` / desconocido), `region_team`, `pos`, `age_s`.

**Las features sólo leen `observed`** (`app/dataset.py`, versión `f2`). Un
enemigo en niebla aporta su último avistamiento y la antigüedad de ese
avistamiento —información pasada— o nada. Todo lo que entra en features usa
`trajectory.pos_at_causal` (sólo muestras anteriores al instante), porque
`pos_at` coge la más cercana y puede ser del futuro.

`scripts/check_leakage.py` verifica ambas cosas sobre el corpus real y debe
dar 0 fugas y 0 posiciones del futuro.

## Builds de los 10

Reconstrucción por eventos de Match-V5 (`ITEM_PURCHASED/SOLD/DESTROYED/UNDO`),
con las trampas conocidas documentadas en `app/inventory.py`:

- eventos con `participantId <= 0` se ignoran (quirk observado en datos reales);
- línea del ítem de support: se otorga al inicio y sube por `ITEM_DESTROYED`
  (cadena 3865→3866→3867) — aproximación, el tier final llega por su evento;
- `ITEM_UNDO`: quita `beforeId`, devuelve `afterId`; los componentes vuelven en
  silencio (aproximación documentada).

Contraste empírico: la playerlist capturada cada 8s en `pass_omni` ahora guarda
también los ítems; `process_game` reporta la tasa de discrepancia, y el test de
regresión compara el inventario final reconstruido contra `item0..item6` de
Match-V5.

**Pendiente (hallazgo del owner, 2026-09-18): el marcador (Tab) muestra la
build de los diez en ese instante, incluso de los que están en niebla.** Es
verdad de terreno independiente de la reconstrucción por eventos y resolvería
justo donde ésta falla (botas de tier 3 y finales de support se otorgan SIN
evento de compra: por eso 2 de 10 inventarios no cuadran). `interfaceScoreboard`
es una propiedad de `/replay/render`, así que se puede activar por API sin tocar
el teclado; el coste es un clasificador de iconos de ítem (ddragon) sobre la
región del marcador. Planificado como mejora de `app/inventory.py`, no como
sustituto: Match-V5 sigue siendo la fuente reproducible offline.

**Pendiente (hallazgo del owner): los marcos laterales muestran quién está
muerto en cada frame**, sin depender de la playerlist (que solo se muestrea
cada 8s) ni de la niebla. Mejora el filtro de vivos que usan el detector y los
controles de visibilidad.

**Pendiente (hallazgo del owner, 2026-09-19): con el marcador abierto, la
tecla `X` muestra el ORO de los diez.** Sería la solución directa a la
limitación "el oro no es exacto" (hoy sale del frame de 60s) y a la necesidad
más dura de build_tracker (oro de cartera en el instante de la compra).

Sin verificar todavía, y **no se puede automatizar como el resto**: el cliente
**ignora la entrada de teclado sintética** (`keybd_event` no produce ningún
cambio en pantalla con el juego enfocado), aunque sí acepta los clics de ratón
sintéticos — que es como `set_fog` maneja el desplegable de niebla. Receta para
comprobarlo cuando toque:

1. abrir una replay y dejar el juego en primer plano (sin robarle el foco);
2. el owner mantiene Tab y pulsa `X`;
3. capturar la pantalla en ese momento (`scripts/` no tiene aún utilidad para
   esto; sirvió una captura rodante que guarda los frames que cambian);
4. mirar **dónde** aparece el número y si es oro actual o total acumulado.

Si el oro sale y es el actual, hace falta OCR de dígitos sobre esa región —
mucho más simple que clasificar iconos de ítem, y el contraste contra
Match-V5 sirve de control.

**Velocidad de reproducción:** el owner no consigue pasar de 8x en la replay
(probado el 18 y el 19). No se ha medido si la Replay API acepta más por su
cuenta; en todo caso el cuello de botella mayor está en el detector (búsqueda
global por campeón y frame), no en la reproducción.

## Punto de decisión y snapshot

Definidos en `LEXICON.md`. Cada punto de decisión produce un registro en
`decisions.jsonl` con el snapshot completo:

- reloj, POV (pid/campeón/equipo), token de decisión y cadena de segmentos;
- estado de los 10: región, posición, nivel, CS, oro (frame 60s previo,
  marcado `_prev_frame`: legal pero rancio, disciplina build_tracker),
  ítems reconstruidos y su valor en oro (precios de Data Dragon);
- campamentos vivos/muertos (rombos), dragones/larvas/heraldo/barón tomados y
  ventanas de respawn observadas (kill + constante de respawn);
- visibilidad: `info_pct` agregado + crudos por enemigo (si hay pasadas);
- referencias al minimapa: frame más cercano (≤5s) de cada perspectiva
  (`pass_omni/mm_xxxxxx.png`, `pass_blue/...`, `pass_red/...`).

El snapshot referencia los PNG, no los copia: las tres perspectivas ya son los
recortes reales capturados.

## Corpus y dataset

```
data/corpus/<MATCH_ID>/
  mv5_match.json  mv5_timeline.json
  pass_omni/  pass_blue/  pass_red/     (mm_*.png + samples.jsonl + meta.json)
  traj.jsonl  camps.json  visibility.json
  segments.jsonl  decisions.jsonl       (ambos junglas: 2 POV por partida)
  meta.json                             (parche, etapas, QA)
```

`scripts/build_dataset.py` congela un split temporal 70/15/15 **por partida**
(`data/ml/split.json`, inmutable salvo `--rebuild-split`) y exporta
`decisions_{train,val,test}.jsonl`. Nunca se mezclan versiones de lexicón.

## Modelo v1 y evaluación

- Baselines obligatorios: clase mayoritaria; farmear el cuadrante más cercano.
- Modelo básico: `HistGradientBoostingClassifier` (sklearn) sobre features
  estructuradas (`app/dataset.py`, versión `f1`). Nada de imágenes en v1: los
  PNG quedan para el revisor y para un futuro CNN.
- Métricas: top-1, top-3, macro-F1, matriz de confusión, por clase.
- **Ningún número reportable con < 30 partidas** (`MIN_GAMES_REPORT`). El
  entrenador se niega; `--smoke` solo valida fontanería con datos sintéticos y
  lo dice en el artefacto.
- Informes en `data/reports/` con `<métrica, split, n, ruta>` o no existen.

## Bucle de feedback del owner

`scripts/review_app.py` (FastAPI local): elige partida y POV, recorre puntos de
decisión — tres minimapas, builds, barra de info_pct, decisión real del
challenger y top-3 del modelo si hay artefacto — y registra veredicto
(`agree/disagree/unclear` + nota) en `data/feedback/feedback.jsonl`
(append-only). Ese archivo es la materia prima para calibrar lexicón v2 y la
métrica de confianza.

## Plan de recogida

Restricciones duras: las replays solo se descargan **en la región del cliente
(EUW) y en el parche instalado** — la ventana de recogida es el parche en curso.

1. `collect_pro_games.py` (API): semillas = **junglas de LEC + Agurin**
   (`data/pros_seeds.json`: riot_ids explícitos + barrido por nombre de las
   300 cuentas challenger EUW) → solo partidas donde la semilla jugó JUNGLA,
   cola 420, parche vigente → `data/queue.jsonl` (dedup) con el POV del pro
   marcado, y mv5_match/mv5_timeline pre-descargados al corpus. Si los pros
   no llegan al objetivo, `--fill-challenger` rellena con junglas challenger
   por LP y el resumen lo declara. (`collect_matches.py` queda como colector
   genérico.) El jungla rival de cada lobby también genera decisiones
   (etiquetado `seed` en meta.json distingue el POV del pro).
2. Por partida: `fetch_replay.py` (LCU descarga y abre) → `capture_game.py`
   (3 pasadas ≈ 11 min de máquina dedicada) → `process_game.py` (≈ 2 min,
   con puertas de QA: tasa de detección/10, validación de niebla, discrepancia
   de inventarios).
3. Objetivo v1: **40 partidas** = 80 POV ≈ 3–5k puntos de decisión.
   Presupuesto: ~9 h de máquina (nocturno) y ~20 GB de disco.

## Riesgos y validaciones pendientes

1. **Control de niebla por API**: `fogOfWar`/`selectionName` deben producir la
   niebla del equipo correcto. Se valida en la primera captura (chequeo de
   fuente); fallback manual documentado. Hasta entonces es la mayor incógnita.
2. **Retratos a escala**: la fase 1 clasificó 10 campeones con retratos locales.
   A escala, los retratos salen de Data Dragon por partida; el umbral (0.87)
   puede necesitar ajuste — la puerta de QA (detecciones/frame) lo delata.
3. **Centinela**: sin rombo; solo se infiere por CS de jungla en río (regla
   nueva, no auditada en fase 1 — auditar en las primeras partidas).
4. **Geometría del minimapa**: escala fija durante la sesión de captura; el
   frame de calibración inicial la registra en `meta.json`. No tocar el HUD.
5. **Caducidad de replays**: descargar la cola pronto; el parche cambia cada
   ~2 semanas.

## Limitaciones conocidas del prototipo (fase exploratoria)

Aplazadas a propósito: son aproximaciones **documentadas**, no fallos ocultos.
Cualquier número que dependa de ellas hay que leerlo con esto delante.

### Oro no es exacto

`gold_total_prev_frame` / `gold_current_prev_frame` salen del frame de 60s
anterior al instante de la decisión, así que pueden ir hasta 60s rancios. Es
legal (nunca se interpola `currentGold` entre compras, ni se usa el frame
posterior), pero **no es el oro que el jugador tenía al decidir**. El sufijo
`_prev_frame` está en el nombre para que no se confunda. `item_gold` sí es
coherente con los ítems reconstruidos, y arrastra las aproximaciones de
`app/inventory.py` (botas de tier 3 y finales de support se otorgan sin evento
de compra: 1-2 de 10 inventarios no cuadran al final de partida).

### Estados de campamento aproximados

`app/camps.py` deduce vivo/muerto del rombo naranja del minimapa con
histéresis, y:

- el estado es `None` hasta la primera lectura firme (aparece como -1 en las
  features, que es lo correcto: desconocido);
- un campeón encima del campamento tapa el rombo, así que esos frames se
  ignoran y el estado se arrastra;
- el **centinela no tiene rombo**: se infiere por CS de jungla en río, regla
  de fase 2 todavía sin auditar;
- los tiempos de respawn no se modelan: sólo se observa el rombo.

### Qué NO está afectado por lo anterior

La separación verdad/observable, el corte causal y la validación del split son
independientes de estas dos limitaciones: se arreglaron por separado y tienen
pruebas propias.

### Oro del HUD: qué se sabe (2026-09-19)

**El oro EXISTE en el HUD y es justo lo que hace falta.** Con el marcador
abierto y pulsando `X`, cada fila pasa de mostrar items a mostrar
`oro_actual(oro_total)` para los diez: p. ej. `92(6457)`, `726(6126)`,
`1117(6522)`. Cartera y acumulado, ambos equipos.

Detalles medidos:

- **Sustituye a los items** en el marcador, no se ven a la vez. No importa:
  los items vienen exactos de la Live Client API (`playerlist`, cada muestra),
  así que API + recorte del marcador dan las dos cosas.
- **El "200 oro" que sale sin pulsar X es la RECOMPENSA (bounty)**, no la
  cartera. No confundirlos.
- **La Live Client API no expone oro en replay**: no hay ninguna clave `gold`
  en `allgamedata` y no hay jugador activo, así que `activeplayer` (que sí
  llevaría `currentGold`) no está disponible.
- **El toggle es de SESIÓN, no se persiste**: tras cerrar el juego (incluso
  con cierre limpio, `WM_CLOSE`) y reabrir la replay, el marcador vuelve a
  items. En `game.cfg` sólo se escribieron los offsets del minimapa.
  Dentro de UNA sesión sí aguanta: una pulsación cubre las tres pasadas.
- **No hay control de ratón para activarlo** (revisado el marcador y el panel
  de replay) y **no hay evento de oro en `input.ini`** que se pueda reasignar
  a un botón del ratón.

**El bloqueo real: el cliente ignora el teclado inyectado.** Probados y todos
sin efecto (0 cambio en pantalla): `keybd_event` (VK), `SendInput` (VK y
scancode), `PostMessage`, `SendMessage`, y `AttachThreadInput` + `SendInput`.
En cambio **el ratón inyectado SÍ funciona** — es como `set_fog` maneja el
desplegable de niebla. Esa asimetría apunta a Vanguard filtrando la entrada de
teclado inyectada.

**RESUELTO (2026-09-19): el owner reasignó la acción al BOTÓN CENTRAL del
ratón.** El ratón inyectado sí llega, así que `capture.ensure_gold_view()` deja
el marcador en vista de oro sin intervención, en cada apertura de replay
(`prepare()`). No se fía de un umbral absoluto: alterna y se queda con la vista
MENOS colorida, que es la de oro — medido 4.3% de píxeles vivos con oro contra
8.6% con items, y verificado idempotente y autocorrectivo (arranque en oro,
arranque en items, y repetición). Sin ese remapeo habría hecho falta una placa
HID por USB, que es el plan B si algún día cambia.

**Plan B si el remapeo deja de valer: un teclado de verdad.** Una placa HID por
USB (Pi Pico / Pro Micro, ~5 €) programada como teclado: el PC le manda una
orden por puerto serie (que no es entrada de teclado, así que no se filtra) y
la placa emite una pulsación `X` a nivel hardware, indistinguible de una real.
Firmware: ~15 líneas con CircuitPython + `adafruit_hid`. Integración: una
llamada en `capture.py` tras abrir la replay, antes de la primera pasada.

Mientras no haya placa: el oro se captura si el owner pulsa `X` una vez por
replay (cubre las tres pasadas), lo que sirve para lotes pequeños pero no para
dejarlo toda la noche.
