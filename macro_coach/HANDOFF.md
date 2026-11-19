# HANDOFF — estado y hallazgos importantes

Última actualización: **2026-09-20**.

Este documento es "dónde está todo ahora mismo y qué hay que saber para
seguir". El contrato de trabajo está en [`CLAUDE.md`](CLAUDE.md), el
vocabulario de decisiones en [`LEXICON.md`](LEXICON.md), el diseño en
[`docs/design.md`](docs/design.md) y el historial de creencias retiradas en
[`docs/history.md`](docs/history.md). Lo de aquí no sustituye a esos: resume.

---

## 1. Estado actual

| | |
| --- | --- |
| Partidas capturadas | **40/40** (3 pasadas), **todas con ORO** (2026-09-19) |
| Oro del marcador (OCR) | **69.250/69.250 celdas legibles (100%)**, 1 bajada de total, error mediana 2 oro |
| Oro por compra | **10.054 eventos**: 77,9% exactos, 22,1% acotados, 0 sin lectura; anclas err_max 10 oro; control de coste UNIVERSAL vía total del marcador: **n=2.266, mediana 0, 96,8% ≤5 oro**, 2 atípicos documentados (history.md nº 24) |
| Lecturas dirigidas (prototipo) | `preflight_buys.py` validado en `EUW1_7986441889`: 440 paradas / 7,1 min, acotadas 119→82, anclas err_max 2,8; filtro de HUD-viejo-tras-seek imprescindible (nº 26). **Escalar antes del cambio de parche** si se quiere: ~5 h de máquina para las 40 |
| Decisiones | **4.059**, todas `snapshot_v: s2` y `lexicon_v: lex1` |
| POVs | 80 (los dos junglas de cada lobby) |
| Corpus en disco | ~15 GB |
| Auditoría de niebla | limpia: 40/40, separación blue-vs-red 28–38 |
| Fugas de omnisciencia | **0** regiones de ocultos, **0** posiciones del futuro, **0** features de oro de la capa de verdad (f3, 2026-09-20) |
| Pruebas | **62 pasando, 1 saltada** |
| Split congelado | 28 train / 6 val / 6 test partidas → 2.781 / 654 / 624 decisiones (**features f3**, regenerado 2026-09-20) |
| Partida excluida | `EUW1_7983109023` (soloq del owner) — fuera del corpus, sólo fixture |

### Baseline real (sólo validación; test sin tocar)

> **Nota (2026-09-19):** estos números son del corpus PREVIO a la tanda de oro
> (4.093 decisiones). Tras recapturar con oro las decisiones son 4.059 y el
> split/modelo estan sin regenerar. Reentrenar es el siguiente paso.
>
> **Nota (2026-09-20):** además, estos números salieron de features **f2**,
> que filtraba oro de la capa de verdad (`gold_total_pf` de los diez y
> `gold_diff_pf` — ningún jugador ve el oro total; history.md nº 23). f3 lo
> retira y usa la estimación observable `gold_diff_obs` = `item_gold_diff` +
> `my_wallet_pf`. **No publicar ni comparar contra estas métricas**: el modelo
> guardado es f2 y hay que reentrenar sobre el export f3. `review_app` ahora
> RECHAZA artefactos de otra versión en vez de rellenar sus features con -1
> en silencio (proponía top-3 de aspecto normal con 11 entradas inventadas y
> los guardaba en el feedback): sirve la revisión sin propuestas y dice por
> qué, hasta que exista un modelo f3.

Informe: `data/reports/20260919_134126_baseline_val.json`

| Métrica | Modelo | Clase mayoritaria | Regla de región |
| --- | --- | --- | --- |
| top-1 | **0,224** | 0,135 | 0,177 |
| top-3 | **0,439** | — | — |
| macro-F1 | **0,121** | — | — |

n = 28 partidas / 2.873 decisiones (train), 6 partidas / 594 decisiones (val).
Visión utilizable en val: 97,6 %.

**Cómo leerlo.** Es imitación: mide cuántas veces la propuesta coincide con lo
que hizo el challenger, **no** que la jugada sea la mejor. Bate a los dos
baselines en top-1. El macro-F1 bajo es el dato honesto: la cabeza de la
distribución se aprende, la cola no — en validación `CUBRIR_BOT` tiene 2
ejemplos, `CUBRIR_TOP` 3, `OBJETIVO_HERALDO` 4. Por debajo de ~15-20 ejemplos
no se puede afirmar nada por clase. Con 6 partidas de validación, los números
son indicativos, no estables.

---

## 2. Hallazgos de captura (lo que hace que funcione)

### Velocidad: 32x, y el límite real es la densidad

Medido en pasadas completas sobre `EUW1_7986655074`:

| vel | min/pasada | frames | densidad | p90 | det/10 |
| --- | --- | --- | --- | --- | --- |
| 8x | 3,8 | 540 | 2,90 s | 3,10 s | 9,5 |
| 16x | 2,1 | 522 | 2,90 s | 3,40 s | 9,9 |
| **32x** | **1,3** | **520** | **2,80 s** | 3,80 s | **10,0** |
| 64x | 0,9 | 341 | **4,50 s** | 6,90 s | 9,7 |

**64x no**: adelgaza la muestra a 4,5 s y por encima de 3,5 s la sensibilidad
de la visibilidad cae del 81,7 % al 50 %. Lo que importa son los **segundos de
partida entre muestras**, no la velocidad.

Para que 32x mantuviera la densidad de 8x hubo que arreglar tres cosas:

1. **escritura de PNG en hilo aparte** — con `imwrite` dentro del bucle, el
   disco lo bloqueaba a rachas y 1 de cada 3 ensayos salía con ~97 % de frames
   duplicados;
2. **conexión HTTPS persistente** al reloj de la replay — `urllib` abría un TLS
   nuevo por frame (~40 ms de los ~114 ms de suelo del bucle); con keep-alive,
   9,3 ms;
3. **no procesar mientras se captura** — con `process_all --jobs 8` ocupando
   los 16 núcleos, a 16x salía el 98 % de duplicados.

El bucle ahora **detecta frames idénticos**: los cuenta, no los guarda y aborta
la pasada si encadena `DUP_STREAK_ABORT`.

### Las tres pasadas: para qué es cada una

`TRUTH-SEEKER` (omni, niebla "Todos") captura minimapa + **marcador en vista de
oro** + playerlist: es la verdad y la única fuente de oro instantáneo.
`REAL-BLUE` y `REAL-RED` capturan minimapa con la niebla de cada equipo.

**El espectador no limita la ficha de un jugador a lo que ve su equipo**
(confirmado por el owner, 2026-09-19): el marcador reparte las builds de los
diez con cualquier niebla. Así que las pasadas de niebla **no** son la fuente
de las builds — son la fuente de la **visibilidad**, que no se puede sacar de
ningún otro sitio. "Lo que veía el azul" se *construye*: build exacta de la
`playerlist` filtrada por esos intervalos de visibilidad. Leer una build de una
pasada con niebla y llamarla "lo que ese equipo veía" sería inventar.

### Niebla: el desplegable del HUD, por color de fila

La Replay API **no** controla la niebla por equipo (`render.fogOfWar` es un
booleano sin perspectiva; `selectionName` es de solo lectura). El control es el
desplegable "Niebla de guerra" del HUD. Su lista se coloca **respecto al valor
seleccionado**, así que los desplazamientos fijos elegían mal: `set_fog()`
localiza la etiqueta, comprueba que la lista se abre y elige la fila **por el
color del texto**.

Esto rompió dos veces de formas que **la validación del momento no ve** (en
t=150 se suelen ver todos los rivales): dos pasadas con la misma niebla, y una
pasada omni con la niebla de la partida anterior. Por eso
**`scripts/audit_fog.py` es puerta obligatoria** antes de procesar: compara
blue contra red en instantes casados (≈30 si son opuestas, 3-6 si son la misma).

### Oro: resuelto con el botón central (2026-09-19)

Con el marcador abierto, la acción de oro cambia cada fila de items a
**`oro_actual(oro_total)`** para los diez: `92(6457)`, `726(6126)`…

- El cliente **ignora el teclado inyectado** (probados `keybd_event`,
  `SendInput` VK y scancode, `PostMessage`, `SendMessage`,
  `AttachThreadInput`: cero efecto). El **ratón inyectado sí llega**.
- **El owner reasignó la acción a mouse3**, así que
  `capture.ensure_gold_view()` la activa sola en cada apertura de replay.
  Ahora **comprueba leyendo las diez celdas** con el OCR, no por lo colorida
  que salga la barra: ese criterio indirecto (4,3 % de píxeles vivos con oro
  contra 8,6 % con items) es justo el que dejó pasar las 40 partidas enteras
  en vista de items sin quejarse. El criterio de color queda de reserva para
  cuando aún no hay plantillas, y la meta de la pasada dice cuál se usó
  (`gold_view.verified_by`).
- El toggle es **de sesión**: no se persiste ni cerrando limpio, por eso hay
  que ponerlo en cada replay.
- El "200 oro" que se ve sin activar nada es la **recompensa (bounty)**, no la
  cartera.
- La Live Client API **no expone oro** en replay (ninguna clave `gold`, y no
  hay jugador activo).

### El oro del HUD **es** el de Match-V5 (2026-09-19)

Comparado jugador a jugador en el mismo instante (`EUW1_7986655074`, t=900):

| | pantalla | Match-V5 |
| --- | --- | --- |
| Renekton | `92(6457)` | 93 / 6458 |
| Shaco | `726(6126)` | 727 / 6127 |
| Kaisa | `620(8850)` | 621 / 8851 |
| Camille | `566(5191)` | 568 / 5193 |

De los diez: siete a 1 de oro, dos a 2, y Zed a 11. En **todos**, el desfase es
idéntico en la cartera y en el acumulado (Zed: 1117→1128 y 6522→6533, los dos
+11), que es justo la forma que tiene un ingreso ocurrido en el hueco entre las
dos lecturas — no un error de medida. La captura es un instante anterior al
frame de Match-V5.

Dos consecuencias: el oro deja de estar 60 s rancio, y el OCR tiene **verdad de
terreno automática cada 60 s** para entrenarse y validarse sin que nadie mire
píxeles.

### OCR del oro: hecho (`app/ocr.py`)

- **Se localiza solo.** En vez de fijar columnas a mano, segmenta las celdas de
  texto del marcador y acepta la única que encaja con `N(N)`. El KDA lleva
  barras, el CS y la visión no llevan paréntesis, la recompensa lleva letras.
  Probado sobre recortes de 845×260 y de 780×280: lee los diez en ambos.
- **Etiquetado humano: una vez, diez celdas** (`data/ocr/sample_gold.json`).
  De ahí salen las plantillas de los doce glifos (`0-9()`).
- **Validación honesta**: leave-one-out glifo a glifo (se aprende sin él y se
  le pide leerlo) **88/88**, margen mínimo 0,198 sobre la segunda opción.
- **Control negativo a escala**: las 40 partidas del corpus tienen el marcador
  en vista de items — ahí no hay oro que leer. **29.000 celdas, 0 lecturas
  inventadas.**
### Validado a 32x con la replay EN MARCHA (2026-09-19)

`scripts/preflight_gold.py` captura una pasada TRUTH-SEEKER entera a un
directorio aparte y contrasta el oro contra Match-V5. Dos partidas, ninguna de
ellas la que dio las plantillas:

| | `EUW1_7981226110` | `EUW1_7984516268` |
| --- | --- | --- |
| frames a 32x | 652, **0 duplicados** | 641, 0 duplicados |
| recortes de marcador | 205 | 195 |
| celdas legibles | **2050/2050** | **1950/1950** |
| anclas de Match-V5 | 40 (0 ilegibles) | 80 (0 ilegibles) |
| bajadas del oro total | 0 | 0 |
| fuera de horquilla | 0 | 0 |

**El oro se lee entero a 32x.** También quedó verificado el toggle: las dos
partidas abrieron en vista de items (`celdas [0, 10]`) y `ensure_gold_view` lo
corrigió de un clic.

### El HUD va por detrás del reloj de la replay (<1 s)

El recorte tomado cuando la API dice `t` muestra un instante algo anterior.
Medido en `EUW1_7984516268`: de los **4** recortes tomados dentro del primer
segundo tras un anclaje, **3** leen por debajo del oro de ese anclaje; de los
**191** tomados a más de 1 s, **ninguno**. A 32x, <1 s de partida son ~30 ms de
reloj real — uno o dos fotogramas de render.

Para las features es inofensivo (información del pasado, nunca del futuro),
pero **la validación tiene que admitirlo**: `LAG_HUD_S` en `app/ocr.py`. Sin
eso, 30 lecturas correctas de 1.950 salían marcadas como error, una de ellas
por 201 de oro — que es lo que vale un asesinato ocurrido en esa fracción de
segundo, no un dígito mal leído. Las diez celdas de un recorte fallaban a la
vez: un error de OCR habría tocado una, no diez.

### Cómo se valida el oro (dos controles, los dos automáticos)

1. **Horquilla entre anclajes** — el oro total no baja nunca, así que una
   lectura en `t` tiene que caer entre el total del anclaje anterior (con el
   margen del lag) y el del siguiente. Se exige en **todas** las muestras, no
   solo en las que caen junto a un anclaje, y no depende del desfase temporal.
   **Es el criterio del veredicto.**
2. **Error contra el anclaje más cercano** — informativo. Mediana 2-4 de oro,
   pero su p90 y su máximo incluyen el desfase entre la captura y el frame de
   60 s, así que un umbral sobre él mide el reloj, no el OCR.

### No se pueden mezclar pasadas con minimapas de distinto tamaño

Descubierto al recapturar solo la omni de `EUW1_7984516268` (el owner había
devuelto el minimapa a la esquina, 255 px, mientras sus pasadas de niebla eran
de 508 px):

| | omni 508 px (original) | omni 255 px (recapturada) |
| --- | --- | --- |
| azul ve al rojo | 0,747 / 0,137 / 0,563 / 0,88 / 0,585 | 0,909 / 0,327 / **0,968 / 1,0 / 1,0** |
| rojo ve al azul | 0,519 / 0,117 / 0,547 / 0,469 / 0,408 | **4 nulos** + 1,0 |

Los iconos se aprenden en la omni y se buscan en las de niebla: con el doble de
tamaño la correlación se hunde. Lo grave es que **no falla, infla** — la
visibilidad tiende a "siempre visible", que es justo lo que le enseñaría al
modelo que nadie se esconde. El guardia `suspect_no_fog` pilló la pasada roja;
la azul habría pasado con números inventados.

`build_visibility` ahora **se niega** si los tamaños no coinciden. Y de aquí
sale la regla para las tandas: recapturar una sola pasada de una partida vieja
solo vale si el HUD está como estaba; si no, se recapturan las tres.

### Oro EXACTO en el instante de compra (`app/buys.py`, 2026-09-19)

El owner lo pidió claro: para entrenar no vale la cartera de hace 10 s, hace
falta la de EN el momento de comprar. Ninguna fuente lo da sola — Match-V5 da
el instante exacto de cada compra pero la cartera solo cada 60 s; el
TRUTH-SEEKER da la cartera cada ~8 s pero no sabe de compras — así que se
cruzan. Cada compra queda entre dos lecturas de cartera (OCR **y** los frames
de Match-V5, que son oro exacto de los diez y entran como anclas gratis, solo
en TRUTH-SEEKER). En esa ventana el único ingreso continuo es el pasivo
(20,4/10 s, determinista), y la diferencia entre las dos lecturas **mide** el
ingreso real: si coincide con el pasivo, la cartera en la compra es exacta.

Cuando no coincide (CS o un asesinato a mitad de ventana), la regla 7 manda: no
se inventa. Se rescata si un tramo se puede **certificar** de solo-pasivo — y el
único certificado que se acepta es estar MUERTO (bien muestreado; no se farmea
muerto). "Quieto en base" se probó y se descartó: el CS no es un evento del
timeline, así que una oleada farmeada de camino a base entre dos muestras se
colaba (envenenó una ventana con +234 de oro). Lo que no se certifica queda
**acotado** en `[suelo, techo]`, nunca en un punto.

Medido en `EUW1_7984516268` (285 eventos de tienda): **224 exactos**, 61
acotados (intervalo mediano 73 oro), 0 sin lectura. Dos controles independientes
(regla 2b):

- **coste de combinación** contra el gastado medido en pantalla: 43 ventanas,
  error mediana 1 oro, 42/43 dentro de 5. (Aquí saltó un bug real: el coste de
  combinar hay que leerlo de los `ITEM_DESTROYED` del mismo instante, no de la
  receta ddragon — items como 8010 consumen cosas que no son componentes
  literales y la receta cobraba el item entero de más.)
- **anclas de Match-V5**, en una resolución hecha SOLO con OCR (para que las
  anclas no sean a la vez entrada y juez): 13 anclas, error mediana 1,3, **máx
  3,3 oro** (antes de arreglar el certificado de posición, este máx era 851).

En el review (`review_app.py`) cada decisión muestra la última tienda del POV
antes de decidir, con el oro exacto de cada compra. Las compras caen en el
HUECO entre segmentos (el recall no es una decisión de macro), así que se
enseñan por "última tanda antes de t0", no por `[t0,t1]`.

### Calibración del minimapa

Se mide, nunca se asume: correlación multiescala de `data/map_background.png`
(rango 200-680 px). Requisitos: **niebla en "Todos"** (con niebla la
correlación cae a 0,14) y **lejos del final de la replay** (el cartel de
VICTORIA tapa el minimapa). Contornos/Canny y ajuste por mínimos cuadrados
contra Match-V5 **no** sirven (ver `docs/history.md`).

### Otras reglas operativas

- **Lanzar las tandas largas desacopladas** (`Start-Process ... -RedirectStandardOutput`):
  los procesos hijos de la sesión del agente se matan a mitad.
- La replay **cierra el juego al llegar al final**: las pasadas paran
  `END_MARGIN_S` antes y `ensure_replay_open()` reabre.
- Fallos recuperables son `RuntimeError`, nunca `SystemExit`: una ventana
  cerrada cuesta una partida, no la noche.
- **El juego se cae solo de vez en cuando** y no se pelea con ello: cada pasada
  se reintenta 3 veces reabriendo la replay, y `capture_queue.py` da una
  **segunda vuelta** sobre las partidas que fallaron (`--vueltas 1` la quita).
  Como la cola es reanudable, eso sale gratis cuando no hay fallos. Reiniciar
  el PC **no** es el primer recurso: primero probar otra replay, que dice en
  dos minutos si el problema es la máquina o ese `.rofl`.

---

## 3. Hallazgos de datos y metodología

1. **Omnisciencia fuera de las features.** El snapshot separa `players[*]`
   (verdad a posteriori, para etiquetar y revisar) de `players[*].observed`
   (lo que el POV podía saber). Las features (`f2`) sólo leen `observed`. Un
   enemigo en niebla aporta su último avistamiento y su antigüedad, o nada.
   Antes, 312 de 315 enemigos ocultos llegaban al modelo con su región real.
2. **Nada del futuro.** `trajectory.pos_at_causal` sólo mira muestras
   anteriores al instante; `pos_at` coge la más cercana y puede ser posterior.
   `scripts/check_leakage.py` lo verifica sobre el corpus (debe dar 0 y 0).
3. **Desconocido explícito en visibilidad.** Sensibilidad por campeón medida en
   la pasada de su propio equipo; por debajo de `VIS_MIN_SENSITIVITY` (0,50,
   elegido con datos: mediana 0,73, meseta 0,4-0,5) la visibilidad es `None`,
   no "no visible". `info_pct` sólo con ≥3 conocidos. Una pasada sospechosa
   invalida la visibilidad de esa partida.
4. **Split validado.** `frozen_split()` rechaza (`SplitError`) un split con
   partidas excluidas, IDs fuera del corpus, duplicados o versiones viejas. El
   export lo vuelve a comprobar. Antes devolvía el fichero tal cual y coló 100
   filas de la partida propia en train.
5. **Corte temporal único.** `build_snapshot` recibe la serie de campamentos y
   la evalúa en `snapshot_time(t0)`, así el llamante no puede desalinearla.
6. **Todo control necesita su contrario.** El control positivo (el equipo
   observador siempre es visible) daba por buenas unas detecciones con 80-87 %
   de falsos positivos; lo destapó el control negativo (campeón muerto, que no
   está dibujado).
7. **Un control tomado bajo carga mide la carga.** Ver el punto 3 de captura.

---

## 4. Limitaciones conocidas (asumidas en fase exploratoria)

- **El oro de las features sigue saliendo del frame de 60 s** (`*_prev_frame`):
  el OCR ya existe y está validado, pero ninguna partida capturada tiene
  píxeles de oro que leer. Se arregla con datos, no con código.
- **Inventarios**: 1-2 de 10 no cuadran al final (botas de tier 3 y finales de
  support se otorgan sin evento de compra). Los items exactos sí están en la
  `playerlist` cada 8 s — pendiente usarlos en vez de la reconstrucción.
- **Una replay concreta se cae** (`SentryHandleException`, ~45 s tras abrir,
  sin volcado ni evento de Windows: el juego se cierra solo). El 2026-09-19 le
  pasó tres veces seguidas a `EUW1_7986655074`… y **otras dos replays se
  capturaron enteras el mismo día sin un fallo**, así que no es la máquina:
  es ese fichero. No hay que reiniciar el PC. Si vuelve a pasar en varias
  partidas a la vez, entonces sí toca mirar Vanguard y drivers.
- **Campamentos aproximados**: histéresis del rombo, un campeón encima lo tapa,
  el centinela se infiere por CS de jungla (regla sin auditar), sin modelo de
  respawn.
- **Las 40 partidas ya capturadas no tienen oro**: sus recortes del marcador
  son de la vista de items. El oro empieza en la próxima tanda.
- **Cola de clases**: `CUBRIR_*`, `OBJETIVO_BARON` y `OBJETIVO_HERALDO` no
  tienen ejemplos suficientes ni para entrenar ni para evaluar.
- **El 21 % de los campeones queda con visibilidad `null`** (83 de 400, en 32
  de 40 partidas; BOTTOM el peor: 49 %). No es el guardia siendo timorato: es
  el detector fallando de verdad, y la cadena esta medida (2026-09-19, sobre
  `EUW1_7984516268`): las detecciones ddragon de esos campeones caen a 2-3
  radios de icono de su posicion verdadera (Tristana mediana 0,060, Cassiopeia
  0,088, contra 0,03-0,05 de los sanos, medido contra los anclajes de
  Match-V5); el icono se aprende recortando DONDE DICE ESA TRAYECTORIA, asi que
  la mediana sale terreno en vez de icono (visto: el "icono" de Tristana es
  suelo y el de Cassiopeia un marcador rojo); con un icono de terreno la
  sensibilidad da 0,1-0,2 y el umbral lo anula. BOT es el peor porque es la
  zona mas poblada del mapa. **No es de esta tanda**: el baseline entrenado ya
  arrastraba esto. Arreglo propuesto (offline, sin recapturar): re-centrar cada
  recorte antes de la mediana (anillo de equipo o una vuelta de EM), medir con
  la distribucion de sensibilidad y el control negativo de muertos.

---

## 5. Qué hacer a continuación

1. **Una tanda nueva con oro.** Es el cuello de todo lo demás: el OCR está
   hecho y validado, pero no hay un solo píxel de oro capturado. La tanda
   contesta además, sola, si el marcador es legible a 32x (`frac_legible`).
   Antes: diagnosticar la caída de la replay.
2. **Estado de decisión `f3`**: lo que el jugador podía saber y nada más — build
   y runas propias del momento, oro propio exacto, y del enemigo **la última
   build vista** (con su antigüedad), sin oro. La pieza que falta es filtrar la
   build por los intervalos de visibilidad; `snapshot._observed_for` ya hace
   exactamente eso para las posiciones y es donde encaja.
3. **Items desde la `playerlist`** en vez de reconstruirlos por eventos: ya se
   capturan exactos cada 8 s y arreglan justo donde falla la reconstrucción.
3b. **Re-centrado de los recortes al aprender iconos** (ver la limitación del
   21 % con visibilidad `null`): es offline, no necesita recapturar nada, y
   sube la visibilidad justo donde más falta (bot lane). Métrica de éxito: la
   distribución de sensibilidad (`vis_quality.py`) y el control negativo.
4. **Decidir el tamaño del corpus.** 40 partidas dan ~4.000 decisiones y bastan
   para la cabeza de la distribución. Para la cola hacen falta ~70 partidas
   (tokens ≥2 %), ~150 (heraldo) o ~300 (barón, cubrir). Alternativa más barata:
   **colapsar tokens raros en un lexicón v2** (`CUBRIR_*` juntos,
   heraldo+barón como objetivo épico) — es cambio de contrato, requiere visto
   bueno del owner.
5. **Más partidas**: el cuello no es la máquina (a 32x son ~4 min/partida) sino
   la oferta: Agurin y SkewMond suman ~40 partidas de jungla por parche.
   Ampliar semillas en `data/pros_seeds.json` o aceptar challenger en general.
6. **Revisión del owner** con `scripts/review_app.py` para empezar a calibrar
   lexicón y métrica de confianza con feedback real.

---

## 6. Comandos

Ver la tabla completa en `CLAUDE.md`. Los de uso diario:

```
.venv\Scripts\python.exe scripts\status.py            # estado
.venv\Scripts\python.exe scripts\audit_fog.py         # PUERTA antes de procesar
.venv\Scripts\python.exe scripts\process_all.py --jobs 8
.venv\Scripts\python.exe scripts\check_leakage.py     # debe dar 0 y 0
.venv\Scripts\python.exe scripts\vis_quality.py       # sensibilidad
.venv\Scripts\python.exe scripts\build_dataset.py --rebuild-split
.venv\Scripts\python.exe scripts\train_baseline.py
.venv\Scripts\python.exe scripts\review_app.py        # http://127.0.0.1:8010
```

Captura (máquina libre, sin procesar a la vez), desacoplada:

```
Start-Process .venv\Scripts\python.exe -ArgumentList "-u","scripts\capture_queue.py" `
  -RedirectStandardOutput data\logs\capture_<stamp>.log `
  -RedirectStandardError  data\logs\capture_<stamp>.err -WindowStyle Hidden
```
