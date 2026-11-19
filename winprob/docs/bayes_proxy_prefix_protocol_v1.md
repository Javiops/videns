# Closed-prefix census for proxy likelihood evidence

2026-10-03. Before any new proxy forecast or accuracy, retain exactly the2900
original16.17focals from recipe-transition-selection-20261001-v1 (manifest SHA
979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6).
Admit the original4001FIT cache via BaseCache's metadata/reservation validator;
the census queries only the original3518game16.17subset and its2900focals.
Keep618unavailable original focal games disclosed; do not select replacements.

Read identity/episode/start metadata and the bound history-cache closure/event
sidecars (manifest SHA1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b).
Validate every query's original state, episode, source event and cutoff. Publish
query_selection.json with its digest before opening the full-action array.

Only then inspect full-action IDs STRICTLY BEFORE each query state in its own
player episode, and interpret only whether the ID is0 (synthetic no-buy) or
nonzero (real purchase). Remove synthetic no-buy nodes first: their existence
uses a later qualification window, so they must not contribute presence,
absence, order, evidence or count. Keep ward-only real visits. Of the real
visits retain exactly those with visit_end_ms < current_start. Strict ties are
masked. Do not follow the old two-slot adjacency graph to reconstruct the
whole prefix: closure ties at intermediate visits can omit an earlier event
that is already closed by the actual query cutoff.

Output chronological prefix state indices/offsets without action contents,
and counts by fold, prefix-size histogram, zero-history queries, total unique
past forecast rows, excluded synthetic nodes, ties and overlaps. Include every
query regardless of history. No current/future action value is indexed; do not
read main-action IDs, prediction/model payloads, rewards, terminal arrays,
raw archive, database,859development or1911confirmation payloads. State
explicitly that hashing the complete full-action file touches its bytes while
only past scalar kind values are interpreted. This is availability, not gain.

CPU1, no GPU, hidden task with startup2.5GiB/live0.75GiB, no game/competing
protected job;15minute hard worker. No model training or policy selected by
this census. Existing source artifacts/code and original index stay unchanged.
