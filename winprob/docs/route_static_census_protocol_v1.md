# Public route geometry census, no game payloads or scores

Declared2026-10-03 before compiling the real public catalogue's compatibility
matrix. This is a necessary feasibility check for route_marginal_design_v1.md,
not a predictive experiment or a claimed improvement.

Read only three existing versioned public/definition artifacts:

- Full action catalogue SHA14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8.
- Static item/champion metadata SHA5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0.
- Public goal definition SHA4457a015b1d08a0893acfc19b8016b7a7104d98f220f2c893949d055144a9072.

Use native16.18 item recipes, matching frozen M2. There are9333main baskets;
combine the two old NONE statuses into one empty/background forest and retain
all257nonempty counted goal forests from that old PUBLIC catalogue definition.
These are candidate construction routes, not observed intentions. The old
106eligible major-item IDs exclude boots and their descendants by definition;
explicitly report that limitation rather than calling the ontology complete.

Build only the public recipe closure needed by those roots and basket items.
Use listed from edges with multiplicities; ignore into and specialRecipe as
construction rules. Missing referenced items or cycles fail this compiler.
This does not simulate native footwear substitutions, passive transforms,
current inventory, gold, affordability, slots or actual material lineage.

For each basket/forest pair, compute the exact maximum count matched to a
recipe frontier using the bounded dynamic program. Parent and child matches
cannot occupy the same recipe branch. Encode matched fraction in integer
units of1/60 (LCM of basket sizes1..6), with empty basket all0. No game/action
frequencies or current/future purchase labels enter the computation.

Report public catalogue counts, reachable recipe nodes, runtime, matrix bytes,
how many baskets have any match, full match, no match, and exactly equivalent
compatibility rows. Distinct baskets with identical rows cannot be reordered
relative to each other by a route update whose R depends only on these rows.
Give a few public item-ID examples. These counts are ontology geometry, not
purchase frequency, achievable accuracy or a gain ceiling on actual games.

As a SYNTHETIC validation only, use R=softmax(score/60) across258routes, uniform
p0 and a seeded synthetic alpha. Verify the joint-marginal and neutral-update
properties and measure a full9333-class bridge call. This fixes no real alpha
estimator, routing table, policy strength or future-label use for a real test.

No game payload, target/future-label/timing/reward array, raw archive, database,
source runtime,859development or sealed/confirmation read. No training or
M2 inference. CPU1; new hidden one-shot task; startup RAM2.5GiB/live0.75GiB,
no game/competing source job;20minute worker. Time first128public baskets and
require1.5*mean_cost*9333+120s fit80% of remaining before finishing the matrix.
Do not truncate the catalogue after a failed bound. Preserve all versions and
the existing index. A complete real prediction plan remains required afterward.
