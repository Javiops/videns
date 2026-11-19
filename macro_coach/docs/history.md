# history.md — creencias retiradas y trampas que costaron horas

Añadir aquí cada enfoque desmentido, con fecha y evidencia. Leer antes de
proponer nada que suene a atajo.

## Fase 1 (prototipo, 2026-09)

- **Live Client `eventdata` no es fiable entre pasadas**: acumula duplicados
  (llegó a marcar 9 kills donde había 6) y pierde objetivos con los seeks.
  Solo `playerlist` (snapshot de estado) es utilizable. Evidencia: auditoría
  manual de EUW1_7983109023.
- **`WARD_PLACED` llega sin coordenadas** en Match-V5. La visión no puede
  reconstruirse desde la API. Consecuencia fase 2: la visibilidad sale de
  pasadas con niebla; los wards quedan fuera de alcance (decisión del owner,
  2026-09-18).
- **La geometría del minimapa cambió a mitad de sesión** (508px → 414px) y
  rompió el recorte: hay que registrar la geometría por pasada, no asumirla.
- **Un campamento es jungla por definición**: gromp y krugs rozan carriles en
  la máscara real; sin el recorte de discos (r=0.042) las etiquetas de farmeo
  salían como gank/carril.
- **Gank ≠ pisar carril**: sin el ancla de rival cercano (<0.15) o kill, la
  mitad de los "ganks" eran cruces o empujes. Igual con farmear ≠ cruzar
  (rombo muerto o CS de jungla, si no es tránsito).

## Fase 2

### 2026-09-18 — primera captura real: cuatro creencias retiradas

1. **La Replay API NO controla la niebla por equipo.** `render.fogOfWar` es un
   booleano sin perspectiva (con él activo llegaron a verse MÁS campeones que
   sin él) y `selectionName` es de **solo lectura**: devuelve el invocador
   seleccionado ("elijah"), y un POST no lo fija. Las teclas 1-5/6-0 tampoco
   seleccionan jugadores en replay. El control real es el desplegable
   **"Niebla de guerra" del HUD de replay (abajo a la izquierda): Azul / Rojo
   / Todos**, que se maneja con clics. Validado midiendo campeones visibles:
   con niebla azul desaparecen rivales (Elise, Camille) y nunca los propios;
   con niebla roja, al revés (Zed, Renekton, Shaco). Riesgo #1 **resuelto**.
2. **La geometría del minimapa de fase 1 no vale.** En la replay aparece
   **centrado** (x0=698, y0=495) y no en la esquina inferior derecha. Todo el
   recorte de fase 1 capturaba terreno 3D. El tamaño resultó ser el mismo
   (508px), pero no se asume: se calibra.
3. **Detectar el minimapa por contornos/Canny no es fiable**: elige cuadrados
   distintos según el contenido del momento (418 / 462 / 483 px en la misma
   partida, todos estables al repetir y todos falsos). **Ajustar por mínimos
   cuadrados contra las posiciones de Match-V5 tampoco**: los frames son de
   60s y los campeones se mueven entre el frame y la captura (residuos de 30 a
   90px). Lo que funciona: **correlación multiescala del fondo del mapa**
   (`data/map_background.png`), correlación 0.80, y Match-V5 como
   **verificación** (los quietos casan a 1-3px).
4. **El reloj de la Replay API coincide con el de Match-V5** (offset 0
   comprobado: los frames de 60s casan con lo renderizado al pausar ahí).
5. **El detector necesitaba tamaños proporcionales.** Con retratos de ddragon
   (no hay recortes de HUD para 170 campeones) y los tamaños fijos de fase 1
   se detectaban 8/10; con tamaños derivados del minimapa calibrado y umbral
   0.82, **10/10**.
6. **La validación de niebla no puede hacerse en t=0**: los diez están en sus
   fuentes y cualquier umbral de píxeles falla (así abortó la primera noche de
   captura). Se valida a mitad de partida y contando campeones.
7. **Al terminar la replay, el cliente CIERRA el juego** — y con él la Replay
   API, así que la pasada siguiente se queda sin nada con lo que hablar
   (`ConnectionRefused` tras completar la pasada omni). Las pasadas paran
   `END_MARGIN_S` (20s) antes del final y pausan; además `ensure_replay_open`
   reabre la replay por LCU y recalibra si hiciera falta.
8. **El HUD de replay es estado persistente del cliente**: si alguien oculta la
   línea de tiempo o los marcos de jugador, el panel de replay se mueve y los
   clics de niebla caen en otro sitio. `prepare()` impone `HUD_STATE` en cada
   partida.
9. **Los pings del minimapa no molestan**: medido sobre los 540 frames reales
   de una pasada, la detección media es 9.6/10 campeones (el resto son muertos,
   que no salen en el minimapa). No hace falta tocar nada por ellos.
10. **La calibración hay que hacerla con la niebla en "Todos"**. Con la niebla
    de un equipo, medio mapa está oscuro y la correlación contra el fondo cae a
    0.14: `prepare()` pone "Todos" antes de calibrar. Antes colaba porque una
    replay recién abierta arranca en "Todos"; al encadenar pasadas, no.
11. **Un umbral heredado borró a un campeón entero del dataset, sin error.**
    `TRAJ_MIN_SCORE=0.895` venía de fase 1 (retratos del HUD, umbral 0.87). Con
    los retratos de ddragon las puntuaciones dependen del campeón: Shaco tenía
    mediana 0.863 y **sus 511 detecciones caían todas**, así que su trayectoria
    se quedaba en las anclas de 60s y su POV daba 18 decisiones en vez de 68.
    Dos puertas incoherentes (0.82 al detectar, 0.895 al leer). Ahora van
    igualadas y `process_game` avisa si la cobertura de alguien baja de
    `TRAJ_MIN_COVERAGE`.
12. **Medir la visibilidad verificando en la posición predicha mide la
    QUIETUD, no la visibilidad.** Las tres pasadas muestrean por separado, así
    que la posición "conocida" llega con segundos de retraso y al que se mueve
    se le busca donde ya no está: salían laners al 97-99% y junglas al 0-15%.
    Parecía ciertísimo — es justo el número que parece correcto del que avisa
    `CLAUDE.md`. Lo delató Shaco al 0.00% teniendo el 95% de cobertura.
13. **La detección global con retratos de ddragon tampoco sirve para
    visibilidad**: el control negativo (campeón MUERTO, que no está dibujado)
    salía detectado el 80-87% de las veces. El control positivo solo mide
    falsos negativos; sin control negativo, esos números pasaban por buenos.
14. **Lo que funciona: iconos aprendidos de la propia pasada omni**
    (`app/icons.py`, mediana de recortes en posiciones conocidas y aisladas) +
    ventana local. Barrido medido: CCOEFF, disco 0.42, ventana ±26px →
    **sensibilidad ~79%, falsos ~2%** (con ddragon: 80-87% de falsos). Ni la
    búsqueda global ni ventanas mayores mejoran.
15b. **La guarda de geometría no vale con niebla.** Comparar el recorte contra
    el fondo limpio del mapa funciona en la pasada omni, pero con niebla media
    mapa está oscuro y la correlación cae por debajo del umbral aunque el
    recorte sea correcto: abortaba TODAS las pasadas de niebla. Solo se
    comprueba en omni.
15c. **Los clics del HUD se validan, nunca se lanzan a ciegas.** El
    desplegable de niebla a veces no llega a abrirse (el juego se come el
    primer clic tras recuperar el foco) y entonces el segundo clic cae en el
    mundo: la niebla se queda como estaba y la validación lo delata como "la
    niebla azul oculta a los suyos" (era la roja de la pasada anterior).
    `set_fog` localiza la etiqueta del panel, comprueba que la lista se abre y
    se niega a pulsar si no la encuentra.
15e. **El desplegable de niebla coloca su lista respecto al valor
    seleccionado**, así que los desplazamientos fijos de las opciones solo
    valían con "Todos" puesto. Rompió de dos maneras que **la validación del
    momento no puede ver**, porque en t=150 se suelen ver todos los rivales:
    - dos pasadas de la misma partida con la **misma** niebla (2 partidas), y
    - una pasada **omni con la niebla de la partida anterior** (2 partidas).
    Arreglo: tras abrir la lista se **localizan las filas por el color de su
    texto** (Azul en azul, Rojo en rojo) y se pulsa la fila real; medido,
    blue-vs-red pasa de 3-5 a 29 de diferencia y es repetible (0.00).
    El apaño anterior (invertir el mapeo al detectar el fallo) queda solo como
    respaldo si el color no se distingue.

15f. **Lo delató el cociente pareado, no una validación.** Con la niebla
    equivocada, `observado ≈ sensibilidad` para TODOS los campeones (0.603 vs
    0.583, 0.160 vs 0.162...): a cada uno lo ve igual de bien el rival que su
    propio equipo, lo cual es imposible con niebla de verdad. Una métrica de
    visibilidad de un solo número lo habría enterrado.
    Dos hipótesis intermedias fueron falsas y hay que decirlo: "las pasadas no
    tienen niebla" (la tenían: son oscuras) y "los tamaños de minimapa no
    cuadran" (cuadraban). Lo resolvieron los números por campeón.

15g. **`scripts/audit_fog.py` audita el corpus entero en segundos**: brillo de
    cada pasada contra omni (detecta niebla ausente) y, sobre todo, **blue
    contra red en instantes casados** (detecta la misma niebla dos veces). La
    diferencia es bimodal: ~30 si las nieblas son opuestas, 3-5 si son la
    misma. Pasarlo SIEMPRE antes de procesar un lote.
15d. **Una replay que parece mala puede no serlo.** `EUW1_7986529933` cerró el
    cliente tres veces y se dio por corrupta; capturó perfecta en cuanto se
    dejó de cambiar la niebla a toda velocidad en las pruebas. El culpable era
    el cacharreo, no el fichero.
16. **La sensibilidad restante se corrige con la pareja de pasadas.** No es
    uniforme (depende de cuánto se mueve cada campeón), así que se mide POR
    CAMPEÓN en la pasada de SU equipo —donde siempre es visible— y la
    visibilidad es el cociente `visto_por_el_rival / visto_por_los_suyos`. El
    sesgo de movilidad se cancela porque numerador y denominador son del mismo
    campeón. Resultado en la primera partida: laners 36-82%, junglas 7-17%.

### 2026-09-19 — revisión del owner: cuatro fallos que el pipeline no veía

Los cuatro estaban en código que "funcionaba": nada fallaba, los números
salían y parecían razonables.

16. **Omnisciencia en las features.** `snapshot.py` rellenaba la región de los
    diez desde la trayectoria de la pasada SIN niebla, y `feature_vector`
    mandaba al modelo la región de los cinco enemigos aunque estuvieran en
    niebla: **312 de 315 posiciones ocultas conservaban región** en
    EUW1_7986655074. Un modelo entrenado así aprende con información que el
    jugador no tenía, y su acierto no significa nada.
    Arreglo: el snapshot separa `players[*]` (verdad a posteriori, para
    etiquetar y revisar) de `players[*].observed` (lo que el POV podía saber);
    las features sólo leen `observed`. Enemigo oculto → último avistamiento y
    su antigüedad, o DESCONOCIDO. `scripts/check_leakage.py` lo comprueba
    sobre el corpus real.
17. **Futuro en el snapshot.** `pos_at()` devuelve la muestra más CERCANA, que
    puede ser posterior al instante de la decisión. Servía para reconstruir,
    no para alimentar al modelo. Añadido `pos_at_causal()` (solo muestras
    pasadas, con antigüedad) y usado en todo lo que entra en features.
18. **Split congelado sin validar.** `frozen_split()` devolvía `split.json`
    tal cual: incluía EUW1_7983109023 (partida propia, excluida) y el export
    metió **100 de 220 filas de train** desde ella. Un split congelado sólo
    vale mientras siga describiendo el corpus y el código: ahora se valida
    (excluidas, IDs fuera del corpus, duplicados entre particiones, versiones)
    y se rechaza con `SplitError` en vez de contaminar en silencio. El export
    vuelve a comprobarlo por su cuenta.
19. **"Sin valor" que era un número.** El aviso de sensibilidad insuficiente
    decía que la visibilidad quedaba en `null`, pero el snapshot recibía un
    número igualmente: `TeamVisibility` no conocía la sensibilidad y trataba
    los falsos negativos del detector (~20%) como "no visible". Además
    `visibility.load()` tiraba el `meta` —justo donde vive la sensibilidad— y
    `suspect_no_fog` sólo avisaba.
    Arreglo: desconocido explícito (`known=False`, `conf=None`), umbral
    `VIS_MIN_SENSITIVITY`, `info_pct` sólo si hay ≥ `VIS_MIN_USABLE_ENEMIES`
    conocidos, pasada sospechosa ⇒ visibilidad inutilizable, y `load()`
    conserva `meta`. La corrección agregada `visible_frac_corrected` se queda
    donde debe: auditoría, nunca visibilidad instantánea.
20. **Corte temporal incoherente.** `process_game` pedía los campamentos en
    `t0` mientras el resto del snapshot iba en `t0 - eps`: un campamento que
    moría justo al empezar el segmento aparecía muerto en el estado con el que
    se decidió. Ahora `build_snapshot` recibe la serie y la evalúa en su
    propio instante causal (`snapshot_time()`), así el llamante no puede
    desalinearla. Prueba de frontera en `tests/test_snapshot_timing.py`.

**Lo transversal:** la auditoría de niebla del 19-09 pasó en las 40 capturas y
eso NO valida el detector de campeones — son controles distintos. Un pipeline
puede estar "verde" y seguir entrenando con información imposible.

21. **La Replay API acepta y cumple 16x y 32x — y el render tambien.**
    Medido: el reloj avanza 8.0x / 16.0x / 32.2x reales. A densidad de captura
    real y con la maquina libre, **0% de frames duplicados** a las tres
    velocidades. 16x adoptado (mitad de tiempo de captura, con margen).
    **La trampa, y casi me la trago**: el primer control dio 98% (16x) y 99%
    (32x) de duplicados y concluí que "el motor adelanta la simulacion sin
    dibujar". Era **contencion de CPU**: estaban corriendo los 8 procesos de
    `process_all` ocupando los 16 nucleos. Repetido en maquina libre: 0%.
    Regla: **no procesar mientras se captura**; si hay que compartir la
    maquina, bajar `--jobs` o volver a 8x. Y medir el entorno, no solo el
    sujeto: un control tomado bajo carga mide la carga.
    Queda ademas la espera del bucle derivada de velocidad y densidad objetivo
    (`SAMPLE_EVERY_GAME_S`) en vez de 0.30s fijos.

22. **32x de captura, validado en pasadas completas** (antes 8x). Medido sobre
    EUW1_7986655074, pasadas enteras y reales, no ráfagas:

    | vel | min/pasada | frames | densidad | p90 | det/10 |
    |-----|-----------|--------|----------|-----|--------|
    | 8x  | 3.8 | 540 | 2.90s | 3.10s | 9.5 |
    | 16x | 2.1 | 522 | 2.90s | 3.40s | 9.9 |
    | 32x | 1.3 | 520 | 2.80s | 3.80s | 10.0 |
    | 64x | 0.9 | 341 | 4.50s | 6.90s | 9.7 |

    **64x no**: adelgaza la muestra a 4.5s, por encima del 3.5s donde la
    sensibilidad de la visibilidad cae del 81.7% al 50%. Lo que importa no es
    la velocidad, es la densidad EN SEGUNDOS DE PARTIDA.

    Tres cosas hubo que arreglar para que 32x mantuviera la densidad de 8x:
    - **escritura de PNG en hilo aparte**: con `imwrite` dentro del bucle, el
      disco lo bloqueaba a rachas y 1 de cada 3 ensayos salía con ~97% de
      frames duplicados (el reloj avanza, la pantalla no se redibuja);
    - **conexión HTTPS persistente** al reloj de la replay: urllib abría un TLS
      nuevo por frame (~40ms de los ~114ms de suelo del bucle). Con keep-alive,
      9.3ms por llamada, y 32x pasa de 3.64s de densidad a 2.96s;
    - **no procesar mientras se captura**: con los 16 núcleos ocupados por
      `process_all --jobs 8`, a 16x salía el 98% de duplicados. Un control
      tomado bajo carga mide la carga.

    Además el bucle ahora **detecta frames idénticos**: los cuenta, no los
    guarda (una copia no es una muestra nueva) y aborta la pasada si encadena
    `DUP_STREAK_ABORT`. Antes esto habría sido invisible.

23. **"Visible para ambos equipos" no incluye el oro total** (retirada
    2026-09-20, features f2 → f3). La nota del 19-09 sobre el marcador bajo
    niebla es del ESPECTADOR: en partida real el Tab enseña builds, nivel y
    CS de los diez, pero el oro total no lo ve ningún jugador. f2 enviaba
    `gold_total_pf` de los diez y `gold_diff_pf` (capa de verdad de Match-V5)
    y `check_leakage.py` no lo veía porque solo vigilaba posiciones. f3 lo
    retira y lo sustituye por la estimación OBSERVABLE que haría el jugador
    (decisión del owner): `item_gold_diff` (builds visibles) + `my_wallet_pf`
    (su propia cartera) = `gold_diff_obs`. El check ahora también vigila el
    oro. Campamentos: el owner decidió dejarlos como están (pasada omni).

24. **Un control que selecciona por su propio criterio de admisión no es un
    control** (retirada 2026-09-20). El control de coste de `buys._valida`
    seleccionaba ventanas `ventana_quieta`, cuya admisión ya exige
    `|cartera − pasivo − coste| ≤ TOL_QUIETA`: medía la misma cantidad con el
    signo cambiado y por construcción `err_max ≤ 6` — un precio erróneo
    expulsaba su ventana a `acotado`, fuera del control. Ahora selecciona por
    el certificado de muerte (independiente de la identidad) y A LA PRIMERA
    cazó un bug real: los elixires (2138/2139/2140) se destruyen al comprarse
    en el mismo timestamp y el descuento de combinación los dejaba a coste 0
    cuando la cartera pierde 500. Las ~5 discrepancias de 112–452 que dejó
    visibles NO eran del modelo de precios: eran certificados falsos (ingreso
    real en tramos "muertos", +190/+243 medidos con lecturas dirigidas). La
    versión definitiva (2026-09-20, misma tarde) no asume nada: el TOTAL del
    marcador es oro GANADO —las compras no lo mueven—, así que
    `coste = w1 + (tot2 − tot1) − w2` mide el coste en TODA ventana de una
    compra, sin pasivo ni certificado. Resultado en corpus: n=2.266 (antes
    371), mediana 0, 96,8% a ≤5 oro; los "errores" de 240–452 dieron 1301 vs
    1300, 2497 vs 2500, 1000 vs 1000. Quedan 2/2.266 (>50 oro): una lectura
    de cierre con HUD rancio a 32x (`EUW1_7981226110` pid7 t≈1396: el gasto
    aparece entero en la ventana siguiente) y un ítem de soporte
    (`EUW1_7985589248` pid10, 3114 pagado 314 vs 400). Moraleja doble: un
    control debe seleccionar por algo independiente de lo que mide, y antes
    de modelar un ingreso mira si alguna columna ya lo MIDE.

26. **Lecturas dirigidas por seek: funcionan, pero el marcador puede
    renderizarse VIEJO tras el salto** (2026-09-20, `preflight_buys.py` sobre
    `EUW1_7986441889`). Saltar la replay a tp±1,5 s de cada evento de tienda
    (440 paradas, 7,1 min, solo seeks hacia delante, pausa y foto) rescata
    compras acotadas (119 → 82; exactas 255 → 292 de 374) con anclas
    limpísimas (err_max 2,8 oro). El peligro medido: tras un seek, la pantalla
    cambia y se asienta pero el MARCADOR puede mostrar valores de varios
    segundos antes (totales que BAJAN respecto a la lectura vecina; carteras
    pre-compra 1,5 s después de la compra — una metió 1071 de error). Filtro
    sin modelo de precios, solo física del total (monótono; y sin eventos de
    tienda en medio, Δcartera = Δtotal): descartó 449/4.400 celdas (~10%) y
    el control de coste quedó en mediana 0. Escalar a la tanda requiere el
    parche vivo (las replays mueren al cambiar de parche).

25. **"Muerto en cada muestra interna" no certifica un tramo** (retirada
    2026-09-20). Con ~8 s de cadencia, el extremo quedaba descubierto: vivo
    en `ta`, una oleada de CS (que no es evento del timeline), muerte, y
    todas las muestras internas ya lo pillan muerto — la trampa de "quieto en
    base" (+234 oro) colándose por el extremo. El certificado ahora es
    exacto: muerte fechada al ms (`CHAMPION_KILL` víctima) ≤ `ta`, ninguna
    muerte dentro del tramo, y una muestra cuyo `respawn` restante cubra
    `tb`. Los porcentajes de compras exactas bajaron de forma honesta
    (regla 7: mejor acotado que exacto indebido).
