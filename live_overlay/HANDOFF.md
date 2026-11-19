# HANDOFF — live_overlay

**2026-10-03 (tarde) — UNA SOLA VENTANA.** El owner pidió todo en una única ventana a la izquierda,
con el rótulo VIDENS. y por secciones (WIN, PICK con las 3 mejores y sin Karthus, LANE con oro,
CS/min y KP tú · rival, TEAMS con el daño), sin reproductor de pasos ni cronómetro, en tu turno
como último pick (el resto ya ha elegido). Lo de abajo describe la primera versión, de cinco tarjetas.

**2026-10-03 — WIDGETS DE SELECCIÓN DE CAMPEONES (diseño, `/draft`).** Vista previa de cinco
tarjetas para el draft: WIN (probabilidad BLUE/RED y cuánto movió el último pick), PICK (las cinco
mejores para tu línea con la franja de `/faces`, la mejor en morado, el porqué de la primera; aviso
YOUR TURN), LANE (oro de tu línea en EARLY/MID/LATE, CS/min, KP, kills), TEAMS (oro de las líneas
conocidas, dragones, mezcla de daño) y BAN (lo mejor que puede coger RED, solo en la fase de bans).
Recorre un draft de ranked paso a paso sobre un cliente simulado; los números los da el drafting
tool por HTTP (`../draft`, :8030), sin código compartido. Nunca se muestra la peor respuesta
("worst"), por decisión del owner. Pendiente: que el owner apruebe el diseño; después, leer la
sesión de selección del cliente local (`/lol-champ-select/v1/session`) y abrir las tarjetas como
ventanas, igual que las de partida.

**2026-09-29 (19:55) — MARCAS COMPLETAS: 173 de 173 campeones con su rectángulo en
`config/faces.json` (ninguno en el valor por defecto, ninguno en el formato antiguo de cabeza),
hechas por el owner en `/faces`. Splash propio de Fiddlesticks en `config/splashes/` (el de Data
Dragon era anterior al rework; los demás vienen de CommunityDragon). Pendiente de decidir por el
owner: el tamaño de fila (150x34 en `web/rows.js` y `web/rows.css`); las marcas son fracciones y
sobreviven al cambio. Anchuras marcadas: mín 0,34, mediana 0,60, máx 0,97 del splash.**

**2026-09-29 (18:45) — v0.4: la franja desde la cabeza.** El owner lo aclaró con una imagen: dos
columnas (BLUE, RED) de cinco filas y, en cada fila, la franja horizontal del splash que arranca en la
cabeza del campeón y llega hasta el borde derecho del arte, fundida en negro hacia la cifra. Vuelve un
ajuste manual mínimo: un clic en la cabeza sobre el splash en `/faces` (`config/faces.json`, Yunara
ya marcada a partir de su imagen); la geometría vive en `web/rows.js`, compartida por el panel y la
página. Las cartas de arte de carga de v0.3 y su endpoint se retiran.

**2026-09-29 (18:30) — v0.3: el panel de oro es la pantalla de carga.** Cinco cartas por
equipo con el arte completo de la pantalla de carga de cada campeón (`/img/loading/<id>.jpg`,
Data Dragon, cacheado) y la cifra fundida en negro en la base; muertos en gris, el propio con
borde dorado. Retirados el mismo día: la fila estrecha con recorte, la página `/faces`,
`config/faces.json`, `rows.css` y el endpoint del splash apaisado (ver docs/history.md).
Verificado en Chrome con la partida grabada de diez campeones y en el overlay real.
Suite: 33 tests.

**2026-09-29 (18:05) — v0.2: dos ventanas translúcidas con iconos.** Objetivos
(Barón, dragón, buffs con portadores) y oro (BLUE/RED, icono redondo de campeón por jugador y su cifra; el arte de
splash llegó en v0.3/v0.4). Alfa 150 en reposo y 255 mientras Tab está
pulsada; alto ajustado al contenido; posiciones recordadas en
`data/layout.json`; una instancia por sesión; errores en `data/overlay.log`.
Review adversarial (24 agentes) aplicado: partidas seguidas con el mismo
roster (conflicto de flujo de EventID), nombres repetidos u ocultos,
objetivos solo en el mapa 11, muertos que reviven antes del poll, dragón sin
equipo → spawn `?`, caché de Data Dragon atómica con fallback a versiones
anteriores, catálogo en hilo aparte, poll con guardia y reloj monótono.
Suite: 33 tests. Verificado en la Practice Tool del owner con capturas de
pantalla reales (translucidez, iconos, BLUE/RED); Tab pendiente de que el
owner lo confirme (no se inyecta la tecla).

**2026-09-29 (17:00) — v0.1 creada como proyecto aparte** desde
build_tracker; nada compartido con los hermanos.

## Lo que no funciona y por qué

- `transparent=True` de pywebview y el color clave de
  `SetLayeredWindowAttributes` no valen con WebView2 (el compositor deja el
  fondo del formulario y no filtra el color): solo el alfa uniforme. Por eso
  la ventana es la tarjeta, sin márgenes ni esquinas de CSS.
- pywebview impone `min_size` 200×100 salvo que se le pase otro: sin él, la
  ventana de objetivos no baja de 100 px.

## Cola

1. Owner: confirmar que con Tab pulsada las dos ventanas pasan a opacas.
2. Grabar partidas propias con `scripts/record.py` y rehacer sobre ellas las
   mediciones de `docs/design.md` (spawns, respawns, `price`, nombres).
3. Comprobar en la primera partida real del parche que las duraciones de buff
   (180 s / 150 s) siguen valiendo; el API no las observa.
4. Decidir si los features descartados (campamentos, daño, ultis) entran como
   estimaciones marcadas o quedan fuera (`docs/design.md`).
