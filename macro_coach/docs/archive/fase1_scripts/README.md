# Scripts de fase 1 (superseded, no vinculantes)

Los cinco scripts del prototipo, tal como se auditaron sobre EUW1_7983109023.
Sus rutas apuntan a un scratchpad temporal que ya no existe: se conservan como
referencia del comportamiento auditado, no para ejecutarse.

Sucesores en `app/`:

| fase 1 | fase 2 |
| --- | --- |
| `capture_pass.py` | `app/capture.py` (3 pasadas + validación de niebla) |
| `analyze_images.py` | `app/detect.py` + `scripts/process_game.py` |
| `build_camp_dots.py` | `app/camps.py` |
| `build_regions.py` | (la máscara `data/region_labels.npy` sigue vigente; regenerarla exigiría portar esto) |
| `macro_v5.py` | `app/segments.py` + `app/lexicon.py` |
