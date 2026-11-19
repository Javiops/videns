"""Ward-free partner on the v4 generation (segment C).

Protocol: docs/experiment_stage2_econ_wardfree_v1.md. The unchanged phase-2
Bf recipe (clean init seed 16, exact full-catalogue CE, micro 32 / effective
256, Adam 1e-3 cosine, 16 epochs, ep16 fixed) on the ward-stripped action
space of the new FIT: the 9,333 groups of stage 1 (group 0 = no main
purchase; a basket left empty by stripping maps there), targets = the group
of each FIT truth. Pilot: 2 untimed warm-up + 12 timed effective batches;
above the declared 8 h bound the segment stops (exit 3, reason published).
Scored on the cohort against R1 (the new Bf), paired by game.
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from stage2_newgen import (OUT_DIR, setup, slice_batch, range_batch, decode_batch, reference_rows, pilot,
                           bound_check, compare, beats, write_report, progress_line, now_utc, v1, p2, sha256_file)
from basket_vocabulary_probe import basket_key
from basket_bf_newgen import basket_tensors

EPOCHS = 16
CHECKPOINTS = (4, 8, 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    S = setup(args.smoke)
    stamp = S.stamp
    epochs = 1 if args.smoke else EPOCHS
    baskets_wf = [basket_key(sorted(S.group_main[g].items())) for g in range(1, S.n_groups)]
    cat_idx_wf, cat_cnt_wf, n_items_wf = basket_tensors(baskets_wf, S.n_groups, S.device)
    targets_wf = S.gid[S.fit_targets]
    print(f"setup listo ({S.setup_seconds}s): fit {S.n_fit}, cohorte {S.n_val}; acciones ward-free {S.n_groups} "
          f"(SAVE {int((targets_wf == 0).sum())}/{S.n_fit})", flush=True)
    nll = nn.CrossEntropyLoss()
    arch_wf = (len(S.champ_index), len(S.item_index), len(S.label_index), len(S.item_index) + 1, n_items_wf)

    def make_model():
        torch.manual_seed(16)
        return v1.BasketNet(*arch_wf).to(S.device)

    def make_step(model, opt):
        def step(idxs_all):
            opt.zero_grad()
            for m0 in range(0, len(idxs_all), p2.MICRO):
                idxs = idxs_all[m0:m0 + p2.MICRO]
                state = model.state(slice_batch(S.fit_ds, idxs), S.device)
                vecs = model.baskets.vectors(cat_idx_wf, cat_cnt_wf, torch.arange(S.n_groups, device=S.device))
                loss = nll(model.scores(state, vecs), targets_wf[idxs.to(S.device)])
                (loss * len(idxs) / len(idxs_all)).backward()
            opt.step()
        return step

    report = {"schema": 1, "segment": "C_wardfree", "protocol": "docs/experiment_stage2_econ_wardfree_v1.md",
              "mode": stamp, "created_utc": now_utc(), "recipe": "phase2-Bf full-CE, clean init, 16 epochs, ep16 fixed",
              "epochs": epochs, "n_actions": S.n_groups, "fit_rows": S.n_fit, "cohort_rows": S.n_val,
              "reference_sha256": S.ckpt_sha, "curve": []}
    report_path = OUT_DIR / f"wardfree_report_{stamp}.json"
    # ---- pilot gate ----
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    per_b = pilot(make_step(model, opt), S.n_fit, S.device)
    bound_check(report, "wardfree", per_b, S.n_fit, epochs, report_path, args.smoke)
    del model, opt
    if S.device.type == "cuda":
        torch.cuda.empty_cache()
    # ---- train (recipe fixed) ----
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    step = make_step(model, opt)
    order_gen = torch.Generator().manual_seed(16)
    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(S.n_fit, generator=g)[:10000].to(S.device)
    progress = OUT_DIR / f"wardfree_progress_{stamp}.jsonl"
    t_train = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.monotonic()
        order = torch.randperm(S.n_fit, generator=order_gen)
        for s in range(0, S.n_fit, p2.EFFECTIVE):
            step(order[s:s + p2.EFFECTIVE])
        sched.step()
        fixed = p2.full_ce_on_subset(model, S.fit_ds, targets_wf, subset_idx, cat_idx_wf, cat_cnt_wf, S.device)
        entry = {"epoch": epoch, "fixed_subset_full_ce": round(fixed, 5), "seconds": round(time.monotonic() - t_ep, 1),
                 "utc": now_utc()}
        report["curve"].append(entry)
        progress_line(progress, entry)
        print(f"  wardfree ep{epoch}/{epochs} fixedCE {fixed:.4f} ({entry['seconds']}s)", flush=True)
        if epoch in CHECKPOINTS or epoch == epochs:
            p2.save_checkpoint(OUT_DIR / f"wardfree_new_ep{epoch}_{stamp}.pt", model, opt, sched, epoch)
    ckpt = OUT_DIR / f"wardfree_new_ep{epochs}_{stamp}.pt"
    report["checkpoint"] = {"path": str(ckpt.relative_to(OUT_DIR.parents[1])), "sha256": sha256_file(ckpt)}
    report["train_seconds"] = round(time.monotonic() - t_train, 1)
    # ---- score on the cohort (actions are the groups themselves) ----
    model.eval()
    records = []
    t_eval = time.monotonic()
    with torch.no_grad():
        vecs = model.baskets.vectors(cat_idx_wf, cat_cnt_wf, torch.arange(S.n_groups, device=S.device))
        for s in range(0, S.n_val, 512):
            sl = slice(s, min(s + 512, S.n_val))
            sc = model.scores(model.state(range_batch(S.val_ds, sl), S.device), vecs)
            order = torch.topk(sc.cpu(), min(v1.SHORTLIST, S.n_groups), dim=1).indices
            records.extend(decode_batch(order, S.val_rows[sl], S.group_main, S))
    report["eval_seconds"] = round(time.monotonic() - t_eval, 1)
    p2.dump(OUT_DIR / f"rows_wardfree_new_{stamp}.jsonl", records)
    arms = {"wardfree_new": records, "new_bf": reference_rows(S.n_val)}
    compare(arms, [("wardfree_new", "new_bf")], report, draw_pairs=[("wardfree_new", "new_bf")])
    report["decision"] = {"rule": "practical progress only if wardfree_new beats R1 (new Bf) with the 95% CI excluding 0",
                          "beats_R1": beats(report, "wardfree_new", "new_bf"),
                          "points_vs_R1": round(report["paired"]["wardfree_new_vs_new_bf"]["diff"] * 100, 3),
                          "ci95_points": [round(x * 100, 3) for x in report["paired"]["wardfree_new_vs_new_bf"]["ci95"]]}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_report(report_path, report)
    print(json.dumps({"aggregates": {k: v["first_action_exact"]["value"] for k, v in report["aggregates"].items()},
                      "decision": report["decision"], "movement": report["movement"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
