# Pre-registro: selección del estimador de oro prequential-v3 — congelado 2026-09-14

Parte de la Fase A del plan aprobado (entradas sin fugas). Este documento se
congela ANTES de la primera pasada del arnés. Ningún número de este experimento
toca la validación ni el test; todo ocurre sobre partidas TRAIN de
`16.18-reconstructible-development-v2` y sus timelines raw archivados.

## Pregunta registrada

¿Qué términos de ingreso, añadidos al walk causal `prequential-v2`, reducen el
error de predicción del oro sin aumentar la tasa de sobrecrédito, midiéndolo
contra la única verdad exacta disponible offline (el `currentGold` del
siguiente frame)?

## La regla de admisibilidad (docstring del módulo, gobierna cualquier término)

Una cantidad es admisible si y solo si (a) es medible del stream de eventos
truncado estrictamente antes del primer `ITEM_PURCHASED` de la visita, (b) es
obtenible del estado del cliente live en el instante del `POST
/api/live/session`, y (c) no es función del multiset comprado.

## Mecánica del arnés (`scripts/prequential_gold_harness.py`)

- Una pasada por partida TRAIN: `reconstruct_game(match, timeline, dragon,
  gold_debug=...)` captura `frame_series` y `gold_events` construidos por el
  código de producción. El estimador evaluado es el REAL
  (`app.reconstruct._attach_gold_estimates`) invocado sobre eventos sintéticos
  en t = ts del frame k (k ≥ 1), donde el último frame estrictamente anterior
  es k−1: exactamente la predicción prequential. Verdad = `currentGold` del
  frame k. El walk jamás se re-implementa.
- Los términos candidatos se computan como CORRECCIONES ADITIVAS sobre esa
  predicción, desde los mismos insumos capturados más campos raw de frames
  (goldPerSecond, minionsKilled) y eventos de asistencia — todos con timestamp
  anterior a t. La variante ganadora se implementa después en producción
  (Fase B) y el arnés se re-ejecuta contra la implementación real antes del
  rebuild; una divergencia entre la corrección aditiva y la implementación es
  un fallo de aceptación.

## Particiones anti-autoengaño

- **Fit/report split dentro de TRAIN**: las constantes ajustables se ajustan
  solo en partidas con hash de match_id PAR y se reportan solo sobre las
  IMPARES. Nada se ajusta y reporta sobre las mismas partidas.
- **Estratos**: compra-en-ventana sí/no, kill/asistencia-en-ventana, bucket de
  minuto (0-9/10-19/20+), y valor de `goldPerSecond` del frame base.
- **Regla de selección (congelada)**: gana la variante con menor |error|
  mediano en el estrato SIN compra del split de report, con dos criterios
  duros: (1) la tasa de sobrecrédito de frontera (est > verdad) no sube frente
  a prequential-v2 en ese mismo estrato; (2) el audit de sobrecrédito a nivel
  de visita (abajo) no empeora. El estrato CON compra se reporta solo por
  transparencia: seleccionar ahí es seleccionar en el vecindario de la
  etiqueta.

## Variantes congeladas (y ninguna más en esta pasada)

- **V2** (línea base): el estimador actual sin cambios.
- **A1 — goldPerSecond aditivo**: `+ c_gps × gps(f0) × Δt_s`. gps NO es una
  tasa de ingresos (medido: 0 en ~76% de frames y la cohorte 0 gana más que la
  90); es un término de item de soporte con unidad desconocida; `c_gps` se
  ajusta en el fit split.
- **A2 — oro de asistencia**: `+ c_assist × nº asistencias del pid en
  (f0, t)`, y COHERENTEMENTE `− c_assist × asistencias de la ventana previa`
  dentro del término residual (como v2 ya hace con los bounties del killer),
  para no contar doble. `c_assist` se ajusta en el fit split.
- **A3 — combinaciones A1+A2** (y nada más).
- **CONGELADO, no seleccionable**: `SHOP_DEAD_TIME_MS` — el arnés evalúa en
  fronteras de frame donde no hay parada de tienda; su óptimo para el recorte
  es trivialmente 0 y cualquier conclusión sobre él desde este arnés es
  infundada.
- **Excluidos con registro**: oro de torres (bounty presente 6/142), monstruos
  (0/95), placas (evento sin oro, 32% sin killer) — atribución no fiable.
- La tasa de CS explícita se descarta como variante separada: el residual de
  ventana previa ya captura el farm por construcción (totalGold es ingreso
  acumulado); descomponerlo solo está justificado si A2 demuestra que el
  residual mezcla asistencias, y entonces A2 ya lo corrige.

## Los otros dos informes de la misma pasada

- **Audit de sobrecrédito a nivel de visita** (lee futuro, TRAIN-only, jamás
  alimenta el export): para cada visita real con `gold_est`, cota
  `wallet(t) ≤ currentGold(f_next) + compras en [t, f_next)` (el ingreso es no
  negativo; las ventas son ingreso y se excluyen de la cota). Cuenta de
  visitas con `gold_est > cota + 1`. Línea base v2; se recalcula con la
  corrección aditiva de cada variante.
- **Timing de ventas de apertura** (evidencia comprometida para la Fase B):
  visitas agrupadas por la regla real de 10s (`SHOP_IDLE_MS`), cuántas abren
  con `ITEM_SOLD` conteniendo compra, y si la primera compra es estrictamente
  posterior; distribución del gap. Reproduce con report-path la medición
  ad-hoc de la revisión (798/798 estrictamente posteriores en 150 partidas).

## Qué NO autoriza este documento

Ni cambios de producción, ni rebuild, ni entrenamiento, ni lecturas de
validación/test, ni que el déficit de 77g (subconjunto sesgado por etiqueta)
participe en ningún ajuste. `scripts/validate_gold_est.py` queda en cuarentena
(su referencia es el patrón vetado `gold_left+net_spend`); no se repara.
