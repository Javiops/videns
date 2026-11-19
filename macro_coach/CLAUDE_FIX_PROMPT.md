# Encargo para Claude Code: corregir la validez del macro_coach

Trabaja en `C:\Users\Javier\videns.gg\macro_coach`. Lee primero `AGENTS.md`,
`CLAUDE.md`, `LEXICON.md`, `docs/design.md` y `docs/history.md`. Hay trabajo del
owner sin commitear y un reprocesado paralelo de las 40 partidas en marcha:
conserva ambos; comprueba el estado de esos procesos antes de regenerar ningún
artefacto y evita mezclar salidas producidas por versiones distintas del código.
No modifiques `build_tracker`, `.env` ni claves. No incluyas `data/` en commits.

Quiero que **implementes las correcciones**, no solo que entregues un análisis.
Al revisarlo encontré estos fallos, todavía presentes en el código:

1. **Información omnisciente en las features.** `app/snapshot.py` rellena
   `players[*].region_team` desde `traj` de la pasada sin niebla, y
   `app/dataset.py:feature_vector` envía las cinco regiones enemigas al modelo
   incluso cuando `vision.enemies[*].visible == false`. En la partida
   `EUW1_7986655074`, 312 de 315 posiciones de enemigos marcados como ocultos
   conservaron región. Separa la verdad retrospectiva necesaria para etiquetar
   de las entradas observables al decidir. Para enemigos ocultos, usa solo la
   última posición realmente avistada y su antigüedad, si puede derivarse sin
   mirar el futuro; en caso contrario usa valor desconocido. No sustituyas
   visibilidad faltante por omnisciencia. Comprueba también que `pos_at()` no
   introduzca muestras posteriores al instante del snapshot en features
   destinadas a inferencia.

2. **Split congelado contaminado.** `app/dataset.py:frozen_split` devuelve
   `data/ml/split.json` sin validar los IDs. El split actual incluye
   `EUW1_7983109023`, partida propia excluida en `app/config.py`; el export
   actual contiene 100 filas de esa partida entre 220 filas de train.
   Garantiza que los splits existentes se rechacen si contienen partidas
   excluidas, versiones incompatibles o IDs que ya no pertenecen al corpus.
   Regenera explícitamente el split cuando el corpus y el código estén listos,
   manteniendo la separación temporal por partida y sin consultar test durante
   el desarrollo. Verifica con una prueba que esa partida nunca se exporta.

3. **Visibilidad y controles de calidad.** `scripts/process_game.py` calcula
   sensibilidad por campeón y `visible_frac_corrected`, pero guarda intervalos
   basados en detecciones crudas; `TeamVisibility.info()` trata sus falsos
   negativos como falta de visión. El aviso de sensibilidad insuficiente dice
   que el dato queda en `null`, aunque el snapshot recibe un número. Además,
   `suspect_no_fog` solo avisa, y `visibility.load()` descarta `meta`. Diseña
   una representación explícita de desconocido y un tratamiento coherente de
   sensibilidad; no conviertas una corrección de frecuencia agregada en
   visibilidad por instante sin evidencia. Haz que una pasada sospechosa o un
   detector sin sensibilidad suficiente impida marcar esa visibilidad como
   válida para entrenar. Integra la auditoría de niebla como puerta antes de
   procesar lotes. La auditoría realizada hoy pasó en las 40 capturas; eso no
   valida por sí solo el detector de campeones.

4. **Corte temporal del snapshot.** `process_game.py` llama
   `camps_mod.alive_at(cstates, d['t0'])`, mientras `build_snapshot` fija el
   estado en `t0 - SNAPSHOT_EPS_S`. Alinea campamentos y demás features con el
   mismo instante causal y añade una prueba de frontera donde un campamento
   cambia justo al comenzar el segmento.

Mantén el propósito del proyecto: imitación de decisiones observadas, no
afirmar que el consejo sea la mejor jugada. Si cambias reglas de etiquetado,
sube `LEXICON_V` y regenera las decisiones; si cambias features, sube
`FEATURES_V` y vuelve a exportar. Añade pruebas que fallen con los errores
anteriores y verifica la suite completa. Después de terminar el reprocesado en
curso, reconstruye solo los derivados necesarios de forma coherente. Presenta
un resumen de archivos cambiados, pruebas, auditorías, número de partidas por
split, métricas de validación con `n` y ruta de informe si ya es lícito
entrenar, y limitaciones pendientes. No informes precisión predictiva a partir
del modelo `smoke` ni con menos de 30 partidas según el contrato del repo.
