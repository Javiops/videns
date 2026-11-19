# videns.gg

> [!WARNING]
> **Proyecto descontinuado.** En los parches nuevos Riot ya no deja descargar las replays de partidas de pros, que eran la base de videns. Sin ellas el proyecto no puede seguir funcionando, así que se queda aquí como archivo. El código se puede consultar, pero no se mantiene.
>
> *Discontinued: Riot no longer allows downloading pro players' replays on recent patches, which videns depended on. Kept as an archive; not maintained.*

Herramientas y proyectos de investigación de League of Legends detrás de videns.gg. Cada carpeta es un proyecto independiente con su propio README, `requirements.txt` y venv.

| Carpeta | Qué es |
| --- | --- |
| `build_tracker/` | Asesor de tienda: qué compraban jugadores de high elo en estados de tienda parecidos. |
| `macro_coach/` | Macro-coaching post-partida a partir de la replay y Match-V5. |
| `live_overlay/` | Overlay en vivo con lo que el cliente del juego expone en local. |
| `winprob/` | Investigación de probabilidad de victoria retrospectiva y revisión de decisiones. |
| `draft/` | Réplica y crítica de la receta de win-probability de draft de itero.gg. |
| `brand/` | Logo e iconos (la marca V.). |
| `informes/` | Informes HTML fechados. |

La web está en su propio repo: [Javiops/videns-site](https://github.com/Javiops/videns-site).

## Qué construí y aprendí

Este repo pesa 15 MB. En local, videns ocupa **≈485 GB (452 GiB)** de datos, sin contar entornos virtuales. Esto es lo que hay ahí, lo que se construyó encima y lo que salió de ello.

### De dónde salen los ~485 GB

| Qué | Tamaño | Detalle |
| --- | --- | --- |
| Partidas reconstruidas (SQLite) | 396 GB | 24 bases de datos con **47.751 partidas únicas** de Ranked Solo (EUW + KR, Master/GM/Challenger, parches 16.17–16.19). Solo la principal tiene 40.890 partidas, 3,44 M visitas de tienda y 4,9 M eventos. |
| Datasets de entrenamiento (JSONL) | 33,5 GB | 9 generaciones inmutables, versionadas por hash, con 4,03 M filas de visitas de tienda. |
| Cachés de tensores y arrays de features | 24,5 GB | `.pt`/`.npy` precalculados para entrenar sin volver a procesar. |
| Capturas de replays | 16,4 GB | 71.771 PNG del minimapa: 41 replays × 3 pasadas (macro_coach). |
| Match-V5 crudo (gzip) | 8,6 GB | 50.374 partidas y 44.796 timelines de la API de Riot, más las copias congeladas de cada corpus. |
| Auditoría y snapshots | 4,3 GB | JSON de aceptación de datos, evidencia por partida y snapshots de la Live Client API. |
| Pesos de modelos | 1,2 GB | 203 checkpoints de entre 0,6 y 2,6 M de parámetros. |

Casi todo es dato, no modelo. Pesa tanto por dos motivos:

1. **Cada partida reconstruida ocupa unas 24 veces más que la cruda:** ~1,8 MB en SQLite frente a ~76 KB en gzip. En cada visita a la tienda se guarda una foto del tablero de los 10 jugadores.
2. **Cada partida aparece de media 3,4 veces.** Cada corpus y cada muestra de parche se congela en su propia base de datos, para que cualquier experimento se pueda repetir exactamente sobre los mismos datos.

Tampoco se publicaría aunque cupiera: incluye los PUUID y Riot IDs de unos 20.000 jugadores.

### build_tracker: asesor de tienda

Responde a una pregunta: *¿qué compraron jugadores de high elo en estados de tienda parecidos al tuyo?* Es imitación de conducta, no un optimizador de builds.

- **Recolección.**
  - Cliente propio de la API de Riot (Match-V5, League-V4, Account-V1) sobre las ladders de KR y EUW y una lista de cuentas de pros.
  - Espera 1,3 s entre peticiones, hace como mucho 95 cada 2 minutos y guarda cada respuesta en gzip.
  - Un `CollectorLease` serializa los recolectores, porque la clave de desarrollo caduca cada 24 h y mataba los trabajos a medias.
- **Reconstrucción.** Cada timeline se convierte en *visitas de tienda*: los eventos de ítem dentro de una ventana inactiva de 10 s.
  - Cada visita guarda inventario antes y después, oro, nivel, CS, KDA, el tablero de los 10 jugadores y los objetivos.
  - La etiqueta es el multiset comprado. SAVE significa reaparecer con ≥400 de oro y no comprar en 90 s.
  - Hubo que reconstruir a mano lo que la timeline no cuenta:
    - la quest de support (3865→3867, que no emite `ITEM_PURCHASED`);
    - los costes de combinación recursivos y los undos;
    - las ventas de apertura, que al corregirse bajaron las compras ilegales del 2,75 % al 0,26 %;
    - Magical Footwear, que da las botas sin evento. Reconstruyendo el temporizador se recuperaron 9.706 de 11.555 partidas.
- **Oro causal.** Match-V5 solo da un frame cada 60 s. El oro en el momento de comprar lo calcula un estimador *prequential* que solo mira el pasado, y un test prohíbe leer el futuro.
- **Modelos (PyTorch):**
  - *Baseline:* tabla condicional con backoff.
  - *PrefixModel:* Transformer sobre 10 tokens de jugador (el comprador es el token 0 y el rival de línea va marcado) más un vector de consulta con oro, asequibilidad y runas. Pérdida BCE multi-etiqueta, con cabeza de conteo y cabezas save/plan.
  - *Bf:* codificador Deep Sets de cestas y un puntuador sobre un catálogo de 7.937 cestas posibles.
  - *M2:* mezcla 50/50 de un fine-tune con features económicas y un modelo sin wards.
- **Resultados.**
  - En la generación v3 el modelo acertaba la primera opción el 38,1 % de las veces, frente al 23,0 % de la tabla (191.916 decisiones).
  - En v4, M2 llega al **51,53 % de acierto exacto en la primera opción y al 66,09 % en top-3**, sobre 146.597 decisiones de validación.
  - Tarda 39 ms de mediana y no dio ninguna opción inasequible en 427.940.
- **Producto.** FastAPI, un overlay con pywebview y un exe de PyInstaller (`BuildAdvisor`, 174 MB con torch para CPU). En partida solo lee la Live Client API local y ejecuta el modelo una vez, cuando abres la tienda.
- **Estado.** Ningún modelo se llegó a promocionar. La beta se quedó esperando la Personal API Key.

### macro_coach: macro de jungla a partir de replays

Esta es la parte que vivía de las replays de pros y la que mata el cambio de Riot.

1. **Partidas.** Soloq de junglas pros: un corpus de 20 partidas de Agurin y 20 de SkewMond, del parche 16.18.
2. **Descarga.** El `.rofl` se baja por la API local del cliente (LCU): se crean metadatos para la partida ajena y después se llama a `/rofls/download` y `/watch`.
3. **Captura.** Con la Replay API, cada partida se reproduce **3 veces a 32x**: vista completa, niebla del azul y niebla del rojo.
4. **Visión offline:**
   - calibración del minimapa por correlación multiescala;
   - iconos de campeón aprendidos en la pasada completa;
   - visibilidad de cada equipo, sacada de las pasadas con niebla;
   - campamentos;
   - OCR del oro.
5. **Fusión con Match-V5.** Salen trayectorias cada ~3 s sobre 13 regiones y 4.059 decisiones etiquetadas con un lexicón cerrado de 25 tokens.

**Números de la visión:**
- **OCR del oro:** plantillas de 12 glifos sacadas de 10 celdas etiquetadas a mano. Leyó 69.250 de 69.250 celdas con un error mediano de 2 de oro, y el control negativo dio 0 lecturas inventadas en 29.000 celdas.
- **Visibilidad:** con iconos aprendidos, ~79 % de sensibilidad y ~2 % de falsos positivos. Con los retratos de Data Dragon, los falsos positivos eran del 80–87 %.

**Problemas de ingeniería:**
- **Ir a 32x sin perder muestras.** Hizo falta escribir los PNG en un hilo aparte y mantener una conexión HTTPS persistente (de 40 a 9,3 ms por llamada). A 64x la sensibilidad cae del 81,7 % al 50 %.
- **La niebla no se controla por API.** Se cambia con clics en el desplegable del HUD, eligiendo la fila por el color del texto.
- **Fugas de información.** Se filtraba la región de 312 de 315 enemigos ocultos. Tras corregirlo, `check_leakage` da 0 fugas.
- **El clasificador.** Un HistGradientBoosting de sklearn. Sus cifras de validación se descartaron porque las features usaban oro omnisciente, y no se llegó a reentrenar.

**Por qué muere.** Toda la información densa (posiciones cada ~3 s, visión, oro instantáneo) sale de abrir la replay. Match-V5 solo da un frame por minuto y los wards llegan sin coordenadas. Sin replays nuevas, el corpus no puede crecer.

### winprob: probabilidad de victoria y valor de cada decisión

- **Datos.** 22.388 partidas admitidas por manifiesto con SHA-256. El experimento W1 usa 6.000 partidas y 138.831 estados, con split cronológico fit/calibración/desarrollo.
- **Modelos y evaluación.** Regresión logística, HistGradientBoosting con 19 features, una variante de trayectoria y calibración sigmoide. Se evalúan con bootstrap pareado por partida (2.000 remuestreos) y un gate pre-registrado de ≥0,5 % de mejora en log-loss.
- **Resultado.** El GBM baja el log-loss de 0,5096 a 0,4991 (−2,05 %) con un 75,7 % de accuracy.
  - Añadir la trayectoria (+0,23 %) o 61 features (−0,73 %) no pasa el gate.
  - Las redes con atención o set pooling empeoran el resultado.
- **RL offline.** Un IQL entrenado con 392.448 transiciones exportadas desde build_tracker **pierde contra la imitación simple**: 53,2 % frente a 59,6 %, una diferencia de −6,4 pp con IC [−12,2; −1,3].
- **Mejorar M2.** Ningún intento lo consiguió: memoria de compras, grafo de kills, rutas, proxy bayesiano y destilación.
- **Estado.** Objetivo no alcanzado. El split de confirmación sigue sellado.

### draft: réplica de iTero

- **Datos.** 46.019 partidas: 31.703 de fit, 6.903 de desarrollo y 6.903 de confirmación sellada.
- **Modelo.** Tablas empirical-Bayes calculadas fuera de pliegue más un ensamble de logística y LightGBM. Hay dos versiones: un modelo por perspectiva, como hace iTero, y uno coherente y antisimétrico.
- **Resultado.** El log-loss baja de 0,6933 (solo el lado del mapa) a 0,6805, con un AUC de 0,591.
  - La réplica reproduce la incoherencia de iTero: P(azul) + P(rojo) se desvía de 1 en 1,8 pp de media, y más de 2 pp en el 40 % de las partidas. El modelo coherente lo arregla sin perder acierto.
  - La experiencia del jugador con el campeón es la señal más fuerte.
  - Counters y sinergias se encogen a cero con este volumen de datos.

### live_overlay: overlay en partida

- **Qué es.** Dos ventanas translúcidas con temporizadores de objetivos y oro en objetos por línea.
- **Cómo funciona.** Lee la Live Client API a 1 Hz y usa FastAPI con pywebview y Win32. No tiene modelo, clave de Riot ni base de datos.
- **Hallazgos:**
  - `items[].price` es el coste de combinación, no el total: Filo infinito marca 725 en vez de 3.500.
  - WebView2 solo admite transparencia uniforme.

### Lo que aprendí

- **El fallo típico no es que algo rompa, sino números que parecen buenos.** Pasó varias veces, y una cifra publicada tuvo que retractarse:
  - oro que leía el frame posterior a la compra;
  - features con oro omnisciente;
  - enemigos ocultos que filtraban su región;
  - 153 partidas de test coladas en un snapshot;
  - 6 IDs de test que se cruzaron entre proyectos.
- **Cada control necesita su control negativo.** Un control positivo solo no demuestra nada. Uno tomado bajo carga mide la carga, y uno que selecciona con su propio criterio no controla nada.
- **Los datos y la reconstrucción importan más que la red.**
  - Con un monedero perfecto el techo sube 16,2 pp.
  - Pasar de 500 a 2.000 partidas sube el acierto de 42,6 % a 48,3 %.
  - En cambio, un modelo más grande (d128) empeora aunque su pérdida de entrenamiento sea menor.
  - Los modelos que funcionan tienen entre 0,6 y 2,6 M de parámetros.
- **Pre-registrar gates y sellar el test.** La mayoría de las ideas no los pasaron, y saberlo también es un resultado.
- **Imitar no es optimizar.** Que una compra suba la probabilidad de victoria no la hace mejor. Con n≈200 por celda, el error estándar es de ~2,5–3,5 pp, así que «este ítem hace ganar» no se puede demostrar a esta escala.
- **Congelar todo:** snapshots, manifiestos con SHA-256 y splits cronológicos. Por eso los datos ocupan lo que ocupan.

## Claves

Los `.env` no se suben. Copia el `.env.example` de cada proyecto a `.env` y pon tu clave de Riot.
