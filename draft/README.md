# Draft

Sibling research project: replicate itero.gg's draft win-probability recipe
on the videns.gg match archive and test its weak points (per-perspective
incoherence, "average champion" imputation, mastery handling, draft turns).

Status (2026-10-01): v1 + exploratory v1b + gold v1 complete on the development split; local
draft tool running on the frozen v1 champion-only model;
confirmation split sealed. Summary in `docs/result_v1.md`; protocol in
`docs/protocol_v1.md`; contract in `AGENTS.md`.

## Pipeline

```
scripts/admit.py       metadata-only admission (denies sibling reservations)
scripts/extract.py     raw envelopes -> data/extract/games_v1.jsonl.gz (detached, ~14 min)
scripts/run_v1.py      split, OOF tables, models R0/R1/I/C + ablations, report
scripts/run_v1b.py     exploratory: structure w/o experience, frozen priors, k=50 pairs
scripts/blind_v1.py    T3 partial drafts: avg imputation vs Monte Carlo
scripts/turns_v1.py    model-based turn-order simulation
scripts/turns_value_v1.py  value of seeing the lane opponent before picking
scripts/gold_v1.py     lane gold difference at 12:00 from the draft (docs/result_gold_v1.md)
scripts/extract_frames.py  sidecar gold at 6/12/18/24/30/36 min (data/extract/gold_frames_v1.jsonl.gz)
scripts/gold_v2.py     lane gold at 12/24/30/36 min (docs/result_gold_v2.md); tool phases 12/24/30
scripts/extract_cs_frames.py / profile_v1.py  expected CS at 12/24/30 + team damage mix (docs/result_profile_v1.md)
scripts/start_tool.ps1 local draft tool -> http://127.0.0.1:8030 (scripts/tool_server.py, draft/engine.py, tool/)
```

Use `.venv` (numpy, scikit-learn, lightgbm). Tests:
`.venv\Scripts\python.exe -m unittest discover -s tests`.
