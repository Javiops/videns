# Cross-project reservation omission, 2026-09-29

The first completed W1 pilot (`w1-pilot-20260929-v2`) is quarantined.
Its admission guard read 71 build_tracker split metadata files and excluded
the union of 3,112 test IDs, but did not include macro_coach's independent
split. All six macro_coach test IDs appeared in build_tracker's source-train
population; three were sampled into W1 development. None was used to fit or
calibrate W1. All six were identified by a metadata-only audit after the run
and before the main agent inspected its outcome metrics.

This is a real forbidden payload read, not a hypothetical risk. The run cannot
be presented as preserving every project's test seal. Its files are retained
with a QUARANTINED marker; its metrics are not used for selection or reporting.
The original macro_coach split and code have not been changed.

Correction: the WP project consumes a versioned reservation artifact exported
from macro_coach split metadata. The combined denylist is applied before
database metadata lookups, sampling, or raw archive reads. The manifest binds
the artifact, its originating metadata digest, and all build_tracker split
digests; any change fails closed. A new manifest and fresh extraction/fitting
will establish a clean development run. There is no claim that rerunning
erases the earlier access.

General rule: cross-project evaluation must union the reservations of every
intended consumer, even where the same game is source-train in another project.
