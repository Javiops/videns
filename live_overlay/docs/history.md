# Historia: creencias retiradas y trampas

## 2026-09-29 — nace fuera de build_tracker

Los dos primeros features del overlay (oro de objetos por equipo; buffs de
Barón y Elder con portadores más respawn de épicos) se construyeron primero
dentro de `../build_tracker` (módulo, ruta `/api/overlay` y dos tarjetas en
su panel `/live`). El owner decidió que los overlays de partida en vivo son
cosa aparte del asesor de tienda y su modelo, así que se sacaron a este
proyecto hermano el mismo día, con la regla de macro_coach: nada compartido,
reimplementar. Lo que se reimplementó: la lectura del Live Client API (una
llamada) y el catálogo de objetos de Data Dragon (un `item.json` por versión
en `data/cache/`). En build_tracker no queda nada del overlay; sí quedan en su
`docs/history.md` los dos hallazgos de abajo, porque afectan a su propio
camino en vivo.

## 2026-09-29 — `price` es el coste de combinación; los killer van sin tag

Medido sobre 24 sesiones de snapshots (15 CLASSIC) capturadas del 2 al 4 de
septiembre por otro proyecto y leídas a mano para esta decisión:

1. `items[].price` == `gold.base` de Data Dragon en 707 de 981 objetos, ==
   base == total en los 274 restantes (componentes), nunca == total a solas.
   Filo infinito: 725 frente a 3500. Un tracker que sume `price` infravalora
   una build completa varias veces. Solución: `gold.total` × `count` y `≈`
   para ids fuera del catálogo.
2. `KillerName` / `Assisters` son `riotIdGameName` en 1.630 de 1.630 kills
   atribuidas a campeón; `riotId` y `summonerName` llevan `#TAG` y nunca
   coinciden. Indexar por el id con tag deja todos los objetivos sin equipo.

También medido: primera DragonKill 5:41, mínimo entre dragones 305 s,
primera BaronKill 22:26, mínimo entre Barones 404 s, primera HordeKill 8:14,
primera HeraldKill 15:26. Ningún evento de Atakhan en 15 partidas CLASSIC.
Las duraciones de buff (180 s / 150 s) son wiki: el API no observa buffs.

Corrección del mismo día (review adversarial): "1.630 de 1.630 kills
coinciden con riotIdGameName" era una cuenta parcial. Remedido sobre el
último snapshot de cada sesión: 2.032 kills por riotIdGameName, 23 por
summonerName (jugadores con el nombre oculto: riotIdGameName "", riotId "#",
summonerName igual al campeón) y 10 por torre o súbdito. "summonerName nunca
coincide" era falso; lo que nunca coincide es el riotId con tag.

## 2026-09-29 — la transparencia con WebView2: ni `transparent` ni color clave, solo alfa

Tres intentos hasta la ventana translúcida. (1) `transparent=True` de
pywebview 6.2.1 hace transparente el fondo de WebView2 pero deja pintado el
fondo del formulario WinForms: se ve un rectángulo blanco. (2) Ventana por
capas de Win32 con color clave (`SetLayeredWindowAttributes` +
`LWA_COLORKEY`, página con fondo magenta): la llamada devuelve True pero el
compositor de WebView2 no pasa sus píxeles por el filtro de color y el
magenta se ve tal cual. (3) La misma ventana por capas solo con `LWA_ALPHA`:
funciona, alfa uniforme sobre toda la ventana, verificado con captura de
pantalla real. Consecuencia de diseño: la ventana es la tarjeta (mismo fondo,
sin márgenes, alto ajustado por la página) y las esquinas las redondea DWM.
Dos trampas más: pywebview impone `min_size` 200×100 salvo que se pase otro
(la ventana de objetivos no bajaba de 100 px), y un proceso lanzado como hijo
de la sesión del agente muere al interrumpir la herramienta; desde el agente
se lanza con `Win32_Process.Create` por WMI.

## 2026-09-29 — el arte del jugador: de icono a fila con splash, y de ahí a la carta entera

Cuatro iteraciones en una hora, todas a petición del owner y todas útiles
para saber qué no volver a proponer: (1) icono cuadrado de campeón; (2) fila
estrecha con el splash apaisado de fondo fundido a negro (la cara caía fuera
de la banda en la mitad de los campeones: cada splash coloca al personaje
donde quiere); (3) la misma fila con el arte de pantalla de carga, cuya cara
sí está casi siempre en el mismo sitio, más una página de recorte manual por
campeón con arrastre, píxeles y deslizadores; (4) cartas enteras de la pantalla de
carga, cinco por equipo; y (5), con una imagen del owner que zanjó la
ambigüedad: dos columnas de cinco filas y una franja del splash que arranca
en la cabeza y llega al borde derecho del arte. Lección: cuando la petición
es visual, pedir o proponer una imagen antes de la tercera iteración. El
único dato por campeón es la cabeza (un clic en `/faces`); la geometría de la
franja es una función (`web/rows.js`) y no un recorte libre.
