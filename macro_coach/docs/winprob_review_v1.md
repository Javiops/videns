# Curva retrospectiva de victoria en la revisión

La revisión admite una copia versionada del artefacto de `winprob`. La
inferencia ocurre fuera de macro_coach; esta integración no importa código,
modelos ni corpus de otro proyecto. No modifica snapshots, features, lexicón
ni modelos de imitación.

Copiar el conjunto completo a `data/imports/winprob_curves_v1/`:

- `curves.jsonl` y `curves.sha256`.
- `meta.json` y `meta.sha256`.

El cargador verifica ambos SHA256, el hash del split local, las identidades
permitidas, el modelo declarado y la correspondencia entre puntos y ventanas.
Rechaza probabilidades no finitas/fuera de [0,1], tiempos desordenados,
duplicados, ventanas sin sus extremos observados y artefactos de otro split.
Las rutas del informe de origen son referencias textuales: nunca se abren.
Si falta la copia o no valida, la revisión habitual continúa y muestra que la
curva no está disponible.

Antes de abrir decisiones, metadatos de partida o imágenes, la interfaz
comprueba el split local directamente. Sólo admite train/val y excluye test y
`EXCLUDED_MATCHES`. No llama a `dataset.frozen_split()` ni a `corpus_games()`:
esas funciones pueden abrir partidas para validar el corpus. Los IDs de test
sólo se leen como metadatos de exclusión.

La curva muestra la estimación OMNI del equipo del POV, con un selector para
ver el equipo azul. El marcador utiliza exclusivamente el último punto cuyo
tiempo sea anterior o igual a `snapshot.t`; muestra su antigüedad. No interpola
estados ni usa la muestra futura. Un punto con más de 90 segundos de antigüedad
se señala como antiguo, sin presentarlo como una estimación actual.

Los botones «Descenso estimado» muestran hasta tres ventanas observadas de
unos tres minutos, elegidas por magnitud sin solapamiento, y navegan a la última
decisión anterior al inicio. La duración proviene de los extremos reales;
no se rellena una ventana incompleta. Un descenso describe el estado del
equipo: no identifica un error, una causa ni una alternativa mejor. No hay
métricas por token ni afirmación de utilidad humana medida.

«Estado y validación del modelo» muestra la declaración de desarrollo y los
digests del artefacto importado. La exposición de cada partida al ajuste,
calibración o desarrollo permanece visible junto a la curva. No se presentan
intervalos de confianza inventados para cada punto.

Las pruebas de `tests/test_winprob_review.py` usan únicamente artefactos y
corpus sintéticos, incluido un guardia que impide abrir cualquier archivo de
la partida sellada sintética. Cubren las rutas de índice, decisiones, página,
imágenes, curva y feedback, además de la validación y la selección temporal.
