# Outcome-free context cache for the expanded revision challenge

Keep the exact 4,002 fit and 1,000 calibration games from clean W1 v3. Replace
its development population with all 859 admission-v2 games whose frozen role
is development and patch is 16.19. This selects 5,861 games before raw access.
All historical test/validation IDs, external test reservations and the 1,911
confirmation reservations remain denied. No role or date boundary changes.

Reuse the original x19 arrays and timestamps for every overlapping game:
5,002 fit/calibration plus 303 development games. Verify their exact saved
array equality; this is copy parity, not an independent raw-extraction check.
Only the remaining 556 development games require raw match/timeline access.
Every newly opened source is admitted before its path is constructed and
receives a compressed-byte digest and an access-ledger entry.

The new extractor reads only immutable roster/team, identity/date/patch, frame
state and pre-frame event fields. It never calls the original outcome-reading
metadata helper, reads match winners/final participant statistics, or accesses
the old NPZ y array. Whole DTO deserialization still includes outcome-bearing
fields in memory; none are indexed, copied, printed or used. Synthetic tests
prove x19/timestamp parity with the original extractor and guard forbidden
field access. Existing frame completeness, 3–40-minute range, strict terminal
container exclusion and past-only slope semantics are preserved exactly.

Outputs use schema `winprob-revision-context-v1`, distinguishing a context-only
artifact from a supervised WP cache. NPZ arrays are x, game, role, timestamp_ms,
start_ms, split_time_ms, patch and feature_names; there is no y array. The source
manifest keeps the existing admission, role and compressed-source binding
fields so downstream admission checks can be reused explicitly.

Missing files, malformed sources, zero qualifying states, duplicate identities,
changed reservations/hashes or failed original-array parity abort the run.
No game is silently dropped or replaced. A failed directory retains its status
and ledger but receives no complete cache metadata. Use a new output directory,
one CPU thread, BelowNormal priority, no GPU, a 20-minute internal deadline and
the external durable resource-gated launcher. No model is trained or scored.
