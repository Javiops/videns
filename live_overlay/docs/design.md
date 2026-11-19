# Diseño: qué da el API y qué se puede construir encima

Fuente única de datos: el Live Client Data API que el cliente de League sirve
en `https://127.0.0.1:2999/liveclientdata/allgamedata` durante cualquier
partida (incluida la Herramienta de práctica). Es el endpoint local
sancionado por Riot; aquí no se lee memoria del juego ni se raspa la
pantalla. La única tecla que se toca es Tab, y solo para leer si está
pulsada (`GetAsyncKeyState`), nunca para inyectar entrada.

## Qué expone `allgamedata` (verificado sobre capturas reales, 2026-09-29)

Verificado leyendo 24 sesiones capturadas del 2 al 4 de septiembre de 2026
(15 CLASSIC, 4 KIWI, 3 PRACTICETOOL, 2 CHERRY; parche 16.17), no la
documentación.

| Bloque | Contiene | No contiene |
| --- | --- | --- |
| `activePlayer` | `currentGold` exacto, nivel, runas completas, `championStats` (resistencias, vida, etc.), `abilities` con `abilityLevel` | cooldowns, buffs, posición |
| `allPlayers[10]` | campeón (`championName` visible y `rawChampionName` interno), equipo, nivel, `isDead`, `respawnTimer`, `scores`, hechizos, runas (keystone y árboles), `items[]` con `itemID`, `count`, `slot`, `price` | oro de los demás, posiciones, daño, buffs, cooldowns |
| `events.Events[]` | `GameStart`, `MinionsSpawning`, `ChampionKill`, `FirstBlood`, `Multikill`, `Ace`, `DragonKill` (`DragonType`), `BaronKill`, `HeraldKill`, `HordeKill`, `TurretKilled`, `FirstBrick`, `InhibKilled`, `InhibRespawningSoon`, `InhibRespawned`, `GameEnd`; todos con `EventTime` y, si aplica, `KillerName`, `Assisters`, `Stolen` | campamentos normales, ningún evento de Atakhan en 15 partidas CLASSIC |
| `gameData` | `gameMode`, `gameTime`, `mapName`, `mapNumber` (11 = Grieta) | |

Trampas medidas sobre esas capturas:

1. **`items[].price` es el coste de combinación, no el total.** De 981
   objetos: 707 coinciden solo con `gold.base` de Data Dragon, 274 con base ==
   total (componentes) y ninguno solo con `gold.total`. Filo infinito reporta
   725 frente a 3500. Por eso el valor de inventario sale de Data Dragon
   (`app/ddragon.py`).
2. **`KillerName` y `Assisters` van sin tag.** Son `riotIdGameName` cuando el
   jugador lo tiene (2.032 de 2.065 kills de campeón en el último snapshot
   de cada sesión); los 23 restantes son jugadores con el nombre oculto, que
   llegan como riotIdGameName "", riotId "#" y summonerName igual al campeón,
   y los eventos usan ese summonerName. El riotId con tag nunca coincide.
   Torres y súbditos también aparecen como killer (`Turret_TChaos_L0_P3_…`,
   `Minion_T100L1S64N0391`). Dos jugadores pueden compartir riotIdGameName
   con tags distintos: un nombre ambiguo no se adjudica.
3. **Los EventID reinician a 0 en cada partida.** Con el mismo roster y el
   reloj más avanzado (Practice Tool, revanchas), la partida nueva solo se
   distingue porque un EventID ya visto vuelve con otro evento u hora.

## Lo que hay (v0.2)

### Oro de objetos por equipo

Filas por línea (`position` del API: TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY;
sin posiciones, por valor) para que cada fila enfrente a los rivales de la
misma línea, con un cuadro central de diferencia por línea (flecha hacia el
ganador, azul/rojo, blanco si 0) y otro de equipo. Por jugador, suma de
`gold.total` × `count` de sus objetos. El wallet propio
(`currentGold`) es exacto y viaja en el payload, pero el panel no lo muestra
(decisión del owner: sin línea de pie); el de los demás no existe en el API y
no se estima. Los totales de equipo son solo objetos, para que sean
comparables. Un id fuera del catálogo usa `price` como suelo y se marca `≈`,
también en la línea de ventaja. Dos columnas (BLUE, RED) de cinco filas;
cada fila muestra un rectángulo del splash del campeón con la proporción de
la fila (150x34), el mismo en las dos columnas, fundido en negro conforme
llega a la cifra: en BLUE la cifra a la derecha, en RED a la izquierda con el
fundido al revés. El rectángulo lo coloca y dimensiona el owner en `/faces`
(arrastre, rueda, flechas) y se guarda en `config/faces.json` como centro y
anchura en fracciones del splash; el alto lo fija la proporción de la fila,
así que cambiar el tamaño de fila no invalida las marcas. Las marcas antiguas
de solo cabeza se convierten a la franja desde la cabeza hasta el borde
derecho. Sin marca: rectángulo centrado en (50 %, 35 %) de media anchura
(`web/rows.js`).

### Buffs de Barón y Elder

`BaronKill` y `DragonKill` con `DragonType == "Elder"` dan el instante exacto.
El equipo se resuelve por `KillerName` (o el primer asistente resoluble si el
killer no es un campeón o el nombre es ambiguo; si nada resuelve, el timer
sale con equipo `?`). Portadores: los miembros del equipo vivos en el primer
poll en que aparece el evento, excluyendo a quien el poll anterior mostraba
muerto con un temporizador que cubre el instante de la kill; se quitan al
morir y se identifican por equipo + campeón + id, no por nombre. Si el
overlay se abre más de 5 s después de la kill, los portadores son
desconocidos y el panel lo dice; el timer de equipo sigue siendo exacto
porque la lista de eventos acumula toda la partida.

Duraciones: 180 s Barón, 150 s Elder. Son constantes de la wiki; el API no
expone buffs, así que **no se pueden medir aquí**. Revisar en cambios de
temporada.

### Respawn de Barón y dragón

Barón: aparece a 20:00, reaparece 6:00 tras morir. Dragón: aparece a 5:00,
reaparece 5:00; cuando un equipo llega a 4 dragones elementales, el siguiente
es Elder y aparece 6:00 después (y cada Elder 6:00 tras el anterior). Una
kill de dragón sin equipo resoluble se cuenta aparte; si con ella un equipo
pudo completar el alma, el siguiente spawn se muestra como `?`.

Contraste con las 15 partidas CLASSIC capturadas (cotas, no ajustes):
primera `DragonKill` a 5:41, mínimo entre dragones 305 s (ajustado a 300);
primera `BaronKill` a 22:26, mínimo entre Barones 404 s (compatible con 360,
no ajustado); primera `HordeKill` a 8:14; primera `HeraldKill` a 15:26.
Elder: 2 kills en el corpus, ningún par consecutivo, así que su respawn es
wiki. Solo se muestran objetivos en el mapa 11; en ARAM, Arena o cualquier
otro modo la ventana de objetivos lo dice y la de oro sigue funcionando.

### Imágenes

- Iconos de campeón: Data Dragon `cdn/<versión>/img/champion/<id>.png`. El id
  de imagen es el sufijo de `rawChampionName` (`game_character_displayname_X`)
  comparado sin mayúsculas contra `champion.json` (Wukong → `MonkeyKing`,
  `FiddleSticks` → `Fiddlesticks`); si no coincide, por nombre visible; si
  nada coincide, iniciales.
- Splash propio del owner: `config/splashes/<id>.jpg` manda sobre todo lo demás
  (proporción 1215:717; las marcas son fracciones, así que el tamaño da igual).
- Splash completo (1215x717): CommunityDragon
  `plugins/rcp-be-lol-game-data/global/default/assets/characters/<id>/skins/base/images/<id>_splash_uncentered_0.jpg`
  (id en minúsculas); si ese patrón no existe, la ruta exacta sale de
  `v1/champions/<key>.json` (`skins[0].uncenteredSplashPath`). Data Dragon
  `cdn/img/champion/splash/<id>_0.jpg` solo como respaldo: es un fichero sin
  versión que Riot no siempre actualiza tras un rework (Fiddlesticks sirve aún
  el arte anterior a 2020, comprobado el 2026-09-29).
- Barón y dragones: CommunityDragon
  `game/assets/ux/minimap/icons/{baron,dragon,dragon_elder,…}.png`; Data
  Dragon no publica iconos de monstruos.

Todo pasa por el servidor local, que lo descarga una vez, comprueba que sea
PNG o JPEG y lo guarda en `data/cache/img/`. Sin red y sin caché, el panel
muestra iniciales y el texto del objetivo.

### Ventanas

Dos ventanas pywebview sin marco, siempre encima, que no roban el foco al
crearse. La translucidez es una ventana por capas de Win32 con alfa uniforme
(150/255 en reposo, 255 con Tab): WebView2 no respeta el color clave de
`SetLayeredWindowAttributes` ni el modo `transparent` de pywebview (deja el
fondo del formulario pintado), pero sí el alfa. La ventana es la tarjeta:
mismo fondo, alto ajustado al contenido por la propia página, esquinas
redondeadas por DWM en Windows 11. Las posiciones se guardan en
`data/layout.json` al cerrar.

## Lo que se descartó y por qué

| Petición | Veredicto | Motivo |
| --- | --- | --- |
| Timers exactos de campamentos de jungla | No plausible | No hay eventos de campamentos ni posiciones. El CS del jungler enemigo sube 4 al limpiar un campo pero no dice cuál. Lo que el juego enseña de forma nativa (campos que tu equipo vio morir) no aporta nada nuevo. Solo cabría un estimador probabilístico con un prior de rutas, y habría que llamarlo estimación. |
| Tracker de daño recibido y causado por tipo | No plausible | Ningún dato de daño en el API. Alternativa honesta: perfil esperado del daño enemigo por composición e ítems, cruzado con las resistencias propias de `championStats`. Es una recomendación, no un tracker. |
| Timers de ulti de aliados | No por API; medio por visión | `abilities` solo existe para el jugador propio y sin cooldown. Por visión del HUD se detectaría el casteo y se calcularía el cooldown por nivel e ítems, con error de varios segundos por runas no visibles. Zona gris de la política de terceros de Riot: leerla antes de invertir. |

## Reproducir y grabar

`scripts/record.py` guarda un snapshot cada 2 s en `data/snapshots/` mientras
juegas. `scripts/replay.py --dir <directorio>` sirve el panel reproduciendo
esos ficheros a ritmo real desde el primer poll, sin League abierto. Las
mediciones de arriba se hicieron sobre capturas de otro proyecto leídas a
mano; al tener corpus propio, rehacerlas aquí.
