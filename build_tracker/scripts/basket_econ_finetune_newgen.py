"""Twin economic fine-tune from the new Bf on the v4 generation (segment B).

Protocol: docs/experiment_stage2_econ_wardfree_v1.md. Recipe of
basket_econ_finetune.py unchanged: both arms fine-tune the FULL new Bf
(trunk + basket encoder + scorer) plus a shared zero-init correction head on
the fixed domain D(x) = the frozen parent's cached top-128; the candidate
receives the real standardised economic features, the control receives
zeros; full-catalogue CE recomputed with gradients, micro 32 / effective 256,
Adam 1e-4 cosine, 8 epochs, ep8 fixed, same batch order per arm. Pilot on
the econ arm: 2 untimed warm-up + 12 timed effective batches; above the
declared 12 h bound (both arms) the segment stops (exit 3, reason
published). Both arms are scored on the cohort with the grouped decoder
against R1 (the new Bf): econ vs control is attribution, econ vs R1 is
practical progress.
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from stage2_newgen import (OUT_DIR, setup, slice_batch, range_batch, group_logits, decode_batch, reference_rows,
                           pilot, bound_check, compare, beats, write_report, progress_line, now_utc, v1, p2,
                           sha256_file, load_econ_fit)
from basket_econ_cache import N_ECON
from basket_econ_finetune import TunedNet, LR
from basket_residual_econ import CorrectionNet

EPOCHS = 8
CHECKPOINTS = (4, 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    started = time.monotonic()
    S = setup(args.smoke)
    stamp, device = S.stamp, S.device
    epochs = 1 if args.smoke else EPOCHS
    dom_fit = torch.load(OUT_DIR / f"domains_fit_{stamp}.pt", weights_only=True)
    dom_val = torch.load(OUT_DIR / f"domains_cohort_{stamp}.pt", weights_only=True)
    econ_fit = load_econ_fit(stamp)
    econ_val = torch.load(OUT_DIR / f"econ_cohort_{stamp}.pt", weights_only=True)
    assert dom_fit["parent_sha256"] == S.ckpt_sha and dom_val["parent_sha256"] == S.ckpt_sha
    assert int(dom_fit["n_rows"]) == S.n_fit == econ_fit.shape[0], "FIT domain/features/cache misaligned"
    assert int(dom_val["n_rows"]) == S.n_val == econ_val.shape[0], "cohort domain/features misaligned"
    fit_domain = dom_fit["top_idx"].long()
    val_domain = dom_val["top_idx"].long()
    # Same standardisation contract as the residual experiment (FIT feasible buys, dims 2..9).
    feas = econ_fit[:, :, 1] > 0
    stats_mean, stats_std = torch.zeros(N_ECON), torch.ones(N_ECON)
    flat = econ_fit[feas].float()
    if flat.numel():
        for d in range(2, N_ECON):
            stats_mean[d] = flat[:, d].mean()
            stats_std[d] = flat[:, d].std().clamp(min=1e-4)
    del flat, feas
    torch.save({"mean": stats_mean, "std": stats_std}, OUT_DIR / f"econ_stats_{stamp}.pt")
    sm, ss = stats_mean.to(device), stats_std.to(device)

    def econ_block(raw, arm):
        if arm == "control":
            return torch.zeros(*raw.shape[:-1], N_ECON, device=device)
        return (raw.float().to(device) - sm) / ss

    nll = nn.CrossEntropyLoss()

    def build():
        torch.manual_seed(16)
        base = v1.BasketNet(*S.arch).to(device)
        base.load_state_dict(S.blob["state_dict"])
        return TunedNet(base, CorrectionNet().to(device))

    def make_step(model, opt, arm):
        def step(idxs_all):
            opt.zero_grad()
            for m0 in range(0, len(idxs_all), p2.MICRO):
                idxs = idxs_all[m0:m0 + p2.MICRO]
                scores, _ = model.corrected_scores(slice_batch(S.fit_ds, idxs), device, S.cat_idx, S.cat_cnt, S.n_cat,
                                                   fit_domain[idxs].to(device), econ_block(econ_fit[idxs], arm))
                loss = nll(scores, S.fit_targets[idxs.to(device)])
                (loss * len(idxs) / len(idxs_all)).backward()
            opt.step()
        return step

    g = torch.Generator().manual_seed(16)
    subset_idx = torch.randperm(S.n_fit, generator=g)[:10000]

    def fixed_subset_ce(model, arm):
        nll_s = nn.CrossEntropyLoss(reduction="sum")
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for s0 in range(0, len(subset_idx), 256):
                idxs = subset_idx[s0:s0 + 256]
                scores, _ = model.corrected_scores(slice_batch(S.fit_ds, idxs), device, S.cat_idx, S.cat_cnt, S.n_cat,
                                                   fit_domain[idxs].to(device), econ_block(econ_fit[idxs], arm))
                total += float(nll_s(scores, S.fit_targets[idxs.to(device)]))
                n += len(idxs)
        model.train()
        return total / max(n, 1)

    print(f"setup listo ({S.setup_seconds}s): fit {S.n_fit}, cohorte {S.n_val}, dominio {v1.SHORTLIST}, "
          f"catalogo {len(S.baskets)}, grupos {S.n_groups}", flush=True)
    report = {"schema": 1, "segment": "B_twin_finetune", "protocol": "docs/experiment_stage2_econ_wardfree_v1.md",
              "mode": stamp, "created_utc": now_utc(), "parent_sha256": S.ckpt_sha, "lr": LR, "epochs": epochs,
              "micro": p2.MICRO, "effective": p2.EFFECTIVE, "fit_rows": S.n_fit, "cohort_rows": S.n_val,
              "step0_check": {}, "curves": {}, "checkpoints": {}, "domain_diagnostics": {}}
    report_path = OUT_DIR / f"finetune_report_{stamp}.json"
    # ---- step-0 check (zero correction) + pilot gate on the econ arm ----
    model = build()
    with torch.no_grad():
        _, corr0 = model.corrected_scores(range_batch(S.fit_ds, slice(0, 64)), device, S.cat_idx, S.cat_cnt, S.n_cat,
                                          fit_domain[:64].to(device), econ_block(econ_fit[:64], "econ"))
        max_corr0 = float(corr0.abs().max())
    ce0 = fixed_subset_ce(model, "econ")
    report["step0_check"] = {"max_abs_correction": max_corr0, "fixed_subset_full_ce_parent": round(ce0, 5)}
    print(f"step0: |corr|max {max_corr0:.2e} fixedCE(parent) {ce0:.4f}", flush=True)
    assert max_corr0 == 0.0, "correction not zero at init"
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    per_b = pilot(make_step(model, opt, "econ"), S.n_fit, device)
    bound_check(report, "twin_finetune", per_b, S.n_fit, epochs * 2, report_path, args.smoke)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # ---- train both arms (control first, as in v1), same batch order ----
    progress = OUT_DIR / f"finetune_progress_{stamp}.jsonl"
    rows_by_arm = {}
    for arm in ("control", "econ"):
        model = build()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        step = make_step(model, opt, arm)
        order_gen = torch.Generator().manual_seed(16)
        curve = []
        t_train = time.monotonic()
        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.monotonic()
            order = torch.randperm(S.n_fit, generator=order_gen)
            for s in range(0, S.n_fit, p2.EFFECTIVE):
                step(order[s:s + p2.EFFECTIVE])
            sched.step()
            fixed = fixed_subset_ce(model, arm)
            entry = {"arm": arm, "epoch": epoch, "fixed_subset_full_ce": round(fixed, 5),
                     "seconds": round(time.monotonic() - t_ep, 1), "utc": now_utc()}
            curve.append(entry)
            progress_line(progress, entry)
            print(f"  {arm} ep{epoch}/{epochs} fixedCE {fixed:.4f} ({entry['seconds']}s)", flush=True)
            if epoch in CHECKPOINTS or epoch == epochs:
                p2.save_checkpoint(OUT_DIR / f"{arm}_ft_new_ep{epoch}_{stamp}.pt", model, opt, sched, epoch)
        report["curves"][arm] = curve
        ckpt = OUT_DIR / f"{arm}_ft_new_ep{epochs}_{stamp}.pt"
        report["checkpoints"][arm] = {"path": str(ckpt.relative_to(OUT_DIR.parents[1])), "sha256": sha256_file(ckpt),
                                      "train_seconds": round(time.monotonic() - t_train, 1)}
        # ---- score on the cohort: corrected scores -> grouped decoder ----
        model.eval()
        records, mass_out = [], 0.0
        t_eval = time.monotonic()
        with torch.no_grad():
            for s in range(0, S.n_val, 512):
                sl = slice(s, min(s + 512, S.n_val))
                dom = val_domain[sl].to(device)
                scores, _ = model.corrected_scores(range_batch(S.val_ds, sl), device, S.cat_idx, S.cat_cnt, S.n_cat,
                                                   dom, econ_block(econ_val[sl], arm))
                probs = torch.softmax(scores, dim=1)
                mass_out += float((1.0 - probs.gather(1, dom).sum(dim=1)).sum())
                order = torch.topk(group_logits(scores, S.gid, S.n_groups).cpu(), min(v1.SHORTLIST, S.n_groups), dim=1).indices
                records.extend(decode_batch(order, S.val_rows[sl], S.group_main, S))
        report["domain_diagnostics"][arm] = {"mean_prob_mass_outside_domain": mass_out / S.n_val,
                                             "eval_seconds": round(time.monotonic() - t_eval, 1)}
        p2.dump(OUT_DIR / f"rows_{arm}_ft_new_{stamp}.jsonl", records)
        rows_by_arm[f"{arm}_ft_new"] = records
        print(f"eval {arm}: exact {p2.aggregate(records)['first_action_exact']['value']:.4f}", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    arms = {**rows_by_arm, "new_bf": reference_rows(S.n_val)}
    compare(arms, [("econ_ft_new", "control_ft_new"), ("econ_ft_new", "new_bf"), ("control_ft_new", "new_bf")],
            report, draw_pairs=[("econ_ft_new", "new_bf"), ("econ_ft_new", "control_ft_new")])
    pv = report["paired"]["econ_ft_new_vs_new_bf"]
    pc = report["paired"]["econ_ft_new_vs_control_ft_new"]
    report["decision"] = {
        "rule": "attribution = econ beats its twin control (CI > 0); practical progress ONLY if econ beats R1 (new Bf) with CI > 0",
        "econ_beats_control": beats(report, "econ_ft_new", "control_ft_new"),
        "econ_beats_R1": beats(report, "econ_ft_new", "new_bf"),
        "control_beats_R1": beats(report, "control_ft_new", "new_bf"),
        "econ_vs_R1_points": round(pv["diff"] * 100, 3), "econ_vs_R1_ci95": [round(x * 100, 3) for x in pv["ci95"]],
        "econ_vs_control_points": round(pc["diff"] * 100, 3), "econ_vs_control_ci95": [round(x * 100, 3) for x in pc["ci95"]]}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_report(report_path, report)
    print(json.dumps({"aggregates": {k: v["first_action_exact"]["value"] for k, v in report["aggregates"].items()},
                      "decision": report["decision"], "movement": report["movement"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
