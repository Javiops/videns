# Draft research contract

Read README.md and docs/protocol_v1.md. Work only inside this sibling project.
Never import code from build_tracker, winprob or macro_coach; exchanges are
versioned artifacts with digests.

- Raw archive reads only through `data/admission/admission_v1.json`, which
  denies every sibling project's sealed test / confirmation IDs before any
  payload is opened. Rebuild admission if a sibling adds reservations.
- The confirmation split (`data/split_v1.json`, role `confirm`) is sealed:
  `load_games` refuses it unless `allow_confirm=True`. Open it once, for a
  pre-registered confirmation, never for selection.
- No Riot API calls from this project. PUUIDs are salted hashes
  (`data/salt.hex`, never shared); never commit data, models or salt.
- Report numbers with metric, split, game count and report path. Model-based
  simulations (turn order) are labelled as such and never presented as
  outcome evidence. Associations in tables are not causal pick values.
- One CPU thread, detached `Start-Process` for anything longer than a minute,
  BelowNormal priority; the machine is shared (live servers, other agents).
