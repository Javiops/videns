# M1 diagnostic curve artifact contract

The request exporter reads only macro_coach's frozen split identity metadata
and winprob admission/reservation metadata. It considers macro train/val IDs
only, removes every current global sealed-test ID and every current or prior
winprob confirmation ID, and binds all source hashes. No macro corpus game,
decision or timeline is opened. The request is versioned and never overwritten.

The separate scorer revalidates that exact allowlist before opening any raw
source. It reads only requested, permitted matches from build_tracker's own
existing raw archive. Missing match/timeline files produce explicit unavailable
rows; there is no Riot request. Malformed identity, queue or timeline is a hard
error. No model fitting occurs. The frozen original W1 v3 snapshot, whose
baseline and negative-control gates passed on 998 development games, supplies
the judge. The improved rich model is not substituted.

The scorer feeds only the original 19 state features to the predictor. Outcome
labels returned internally by the existing extractor are never exported or
passed to prediction. All base extraction rules remain: complete nonterminal
frames from minutes 3–40, no future interpolation. Model exposure is disclosed
per game from the judge's actual source cache as fit, calibration, development
or unseen, including in-sample status where applicable.

Output files are curves.jsonl and meta.json, each with a matching SHA256
sidecar (curves.sha256 and meta.sha256). JSONL rows have schema
winprob-macro-curve-v1, match_id, status, model_digest, exposure_role, points,
observed_windows, unavailable_reason, completeness and source_hashes. Points
contain timestamp_ms and p_blue. For each earliest observed clock-minute bin
from 3 through 37 whose +3 bin exists, an overlapping observed window contains
the actual start_ms/end_ms and 100 times the change in blue win probability
as delta_blue_pp. Missing endpoints remain missing; nothing is interpolated.

Metadata schema winprob-macro-curves-v1 binds curves_sha256,
requested_game_ids, macro_split_sha256, model_digest, model_validation,
generated_utc, request/source/code hashes and coverage counts. Model validation
is cohort-level evidence and is explicitly not a per-point interval. Curves
are retrospective OMNI diagnostics; their changes are never attributed to an
action, and no token-level or causal effect is established.

The bounded export uses one CPU thread, at most 34 requested games and at most
150 seconds. It writes only a new winprob artifact directory. Transfer into
macro_coach is a separate authorized copy of versioned artifacts, with no
shared runtime imports or direct model access.
