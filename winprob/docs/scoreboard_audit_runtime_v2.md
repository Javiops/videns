# Scoreboard functional audit: launch-only memory adjustment

Status: NOT APPLIED. Before replacement, the guarded controller found that the
original v1 audit had already started at2026-10-01T11:42:35UTC after RAM became
available. It aborted without stopping any process or changing a launcher.
There is no v2 run or v2 launcher. The original v1 audit completed successfully
at11:45:58UTC and its launcher recorded completion at11:46:05UTC. Preserve the
proposal below as an unexecuted runtime option, not the execution history.

2026-10-01, after the training result was complete. The v1 audit launcher
remained in its resource wait with no Python worker/process receipt or report.
Available RAM was about2.73-2.98GiB, below its inherited3.25GiB start threshold.

The COMPLETED training process reported peak physical working set
1,670,987,776 bytes (about1.56GiB), with minimum observed available RAM
1,455,337,472 bytes. This is measured training-process memory, not a new audit
measurement or a guarantee about concurrent applications.

Replace only the own waiting audit launcher with a new v2 launcher/output.
The audit startup threshold is2.5GiB; live host floor remains0.75GiB and GPU
free requirement3072MiB. The same audit code, checkpoints, batches, exact-replay
tolerances, data, models, metrics and success gates remain unchanged. There is
no fitting, epoch extension or scientific recipe change. No unrelated process
is stopped. Preserve the cancelled v1 wait record and record the controller's
intentional stop separately; never describe it as a model/audit score failure.

The v2 audit must still stop on resource pressure and report actual execution
status. No user rating, permission or account quota is the missing dependency.
