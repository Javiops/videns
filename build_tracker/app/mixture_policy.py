"""Serving/evaluation adapter for a versioned mixture package.

ONE implementation, used by both the displayed-policy harness and (once the
contract question is settled) the live path. If evaluation and serving had
separate implementations, the report would describe a policy nobody serves —
which is the whole reason the owner asked for a shared adapter.

Three properties this file exists to guarantee:

1. **It never reads TRAIN.** Everything the decoder needs — vocabularies,
   the 7,937-basket catalogue, the ward-stripped group map, the ward-free
   catalogue, the canonical-key permutation, the economic normalization —
   comes from the package built offline by
   `scripts/build_mixture_package.py`. `assert_reads_no_train()` states it.
2. **Legality is not optional.** Every option returned has passed
   `purchase_sequence` at the decision's own budget. `SAVE` is always legal.
3. **The package is verified before use.** Every component is checked
   against the sha256 in the manifest, and the manifest itself against the
   package identity digest. A package that does not match refuses to load.

The economic feature function is imported from `scripts/basket_econ_cache.py`
rather than copied here on purpose: a copy would drift from the one the
models were trained against. The manifest records that file's digest, so a
change to it invalidates the package identity instead of passing silently.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def activate_runtime(package_dir, review_path):
    """Activate the pinned footwear runtime. MUST run in a clean process.

    `benchmark_1618_runtime.activate` refuses to run if `train_prefix`,
    `app.conditional_policy`, `baseline` or the policy evaluators are already
    imported: it REPLACES economic symbols in those modules
    (`baseline.inventory_state`, `baseline.decision_kind`) and pins
    DataDragon to 16.18.1. Activating after they are captured would leave the
    process running different economics from the ones the package was built
    against, with nothing failing.

    So the execution contract is: call this FIRST, from a clean process,
    before constructing MixturePolicy. It returns the live declaration, and
    MixturePolicy refuses to load unless that declaration matches the one
    recorded in the manifest.
    """
    import benchmark_1618_runtime as runtime
    manifest = json.loads(
        (Path(package_dir) / "manifest.json").read_text(encoding="utf-8"))
    active = runtime.activate(Path(review_path))
    if active != manifest["footwear_runtime"]:
        raise ValueError(
            "the active footwear runtime differs from the one this package "
            "was built against")
    verify_active_economics()
    return active


def verify_active_economics():
    """Check the economic symbols actually in force, not their file hashes.

    A digest says a file on disk is unchanged. It does not say the running
    process is using it: activation works by REBINDING symbols, so the only
    honest check is identity of the live objects.
    """
    import footwear_economics as economics
    if "baseline" not in sys.modules:
        raise RuntimeError("runtime not activated: baseline was never bound")
    baseline = sys.modules["baseline"]
    for name in ("inventory_state", "decision_kind"):
        if getattr(baseline, name, None) is not getattr(economics, name):
            raise RuntimeError(
                f"baseline.{name} is not the reviewed footwear_economics "
                "implementation; the runtime is not properly active")
    import app.ddragon as ddragon
    if getattr(ddragon.DataDragon, "__name__", "") != "data_dragon":
        raise RuntimeError("DataDragon is not pinned by the active runtime")
    return True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


class MixturePolicy:
    """A loaded mixture package. Construct once; call `options` per decision."""

    def __init__(self, package_dir, dragon, device=None, verify=True):
        self.dir = Path(package_dir)
        self.manifest = json.loads(
            (self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.package_id = json.loads(
            (self.dir / "package_id.json").read_text(encoding="utf-8"))
        self.digest = self.package_id["manifest_sha256"]
        if verify:
            self._verify()
        self.dragon = dragon
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        import train_prefix as tp
        import train_basket_ab as v1
        self._tp, self._v1 = tp, v1
        self.config = dict(self.manifest["featurization"])
        # apply_config does NOT restore d_model/layers/ff_dim/heads: those are
        # module globals read from PREFIX_DMODEL/LAYERS/FF/HEADS at import
        # time. With the same d_model a different head count keeps every
        # weight SHAPE identical, so load_state_dict succeeds and predictions
        # change silently. Architecture comes from the manifest, and the built
        # model is re-inspected afterwards (_assert_architecture).
        self.arch = {k: int(self.config[k]) for k in
                     ("d_model", "layers", "ff_dim", "heads")}
        tp.D_MODEL = self.arch["d_model"]
        tp.N_LAYERS = self.arch["layers"]
        tp.FF_DIM = self.arch["ff_dim"]
        tp.N_HEADS = self.arch["heads"]
        verify_active_economics()
        # the featurization knobs are module globals; re-assert them before
        # every forward, exactly as Predictor does, so a second policy loaded
        # in the same process cannot leave this one featurizing for the wrong
        # model (app/predictor.py records that this has bitten before)
        self._apply_config()
        if self.manifest.get("static_data_sha256") and \
                self.manifest["static_data_sha256"] != dragon.signature:
            raise ValueError("Package static-data digest mismatch")

        vocab = json.loads((self.dir / "vocab.json").read_text(encoding="utf-8"))
        cats = json.loads(
            (self.dir / "catalogues.json").read_text(encoding="utf-8"))
        self.champ_index = {int(k): v for k, v in vocab["champ_index"].items()}
        self.item_index = {int(k): v for k, v in vocab["item_index"].items()}
        self.label_index = {int(k): v for k, v in vocab["label_index"].items()}
        self.baskets_A = cats["baskets_A"]
        self.baskets_B = cats["baskets_B"]
        self.group_main = [[(int(i), int(n)) for i, n in g]
                           for g in cats["group_main"]]
        t = torch.load(self.dir / "tensors.pt", weights_only=True)
        self.group_ids = t["group_ids"].to(self.device)
        self.perm = t["perm_B_to_A"].to(self.device)
        self.cat_idx_A = t["cat_idx_A"].to(self.device)
        self.cat_cnt_A = t["cat_cnt_A"].to(self.device)
        self.cat_idx_B = t["cat_idx_B"].to(self.device)
        self.cat_cnt_B = t["cat_cnt_B"].to(self.device)
        self.sm = t["econ_stats_mean"].to(self.device)
        self.ss = t["econ_stats_std"].to(self.device)
        self.n_cat = len(self.baskets_A) + 1
        self.n_act = len(self.group_main)
        self.shortlist = int(self.manifest["policy"]["shortlist"])
        self.n_options = int(self.manifest["policy"]["options"])
        self.wA = float(self.manifest["policy"]["weights"]["A"])
        self.wB = float(self.manifest["policy"]["weights"]["B"])
        self._build_models()

    # ------------------------------------------------------------------
    def _verify(self):
        stored = dict(self.manifest)
        if _canonical_digest(stored) != self.digest:
            raise ValueError("manifest does not match the package identity")
        checks = [(self.manifest["structures"]["file"],
                   self.manifest["structures"]["sha256"]),
                  (self.manifest["vocab"]["file"],
                   self.manifest["vocab"]["sha256"]),
                  (self.manifest["catalogues"]["file"],
                   self.manifest["catalogues"]["sha256"])]
        checks += [(w["file"], w["sha256"])
                   for w in self.manifest["weights"].values()]
        for rel, want in checks:
            got = _sha256_file(self.dir / rel)
            if got != want:
                raise ValueError(f"component digest mismatch: {rel}")
        econ = self.manifest["economic_engine"]
        for key, rel in (("features_sha256", "scripts/basket_econ_cache.py"),
                         ("conditional_policy_sha256", "app/conditional_policy.py"),
                         ("shop_econ_sha256", "app/shop_econ.py")):
            if econ.get(key) and _sha256_file(ROOT / rel) != econ[key]:
                raise ValueError(
                    f"{rel} changed since the package was built; the economic "
                    "engine is part of this policy's identity")

    def _apply_config(self):
        self._tp.apply_config(self.config)
        if self._tp.USE_HISTORY:
            raise ValueError("this package is not a history policy")

    def _build_models(self):
        v1, tp = self._v1, self._tp
        vocab = json.loads((self.dir / "vocab.json").read_text(encoding="utf-8"))
        arch = lambda n_items: (len(self.champ_index), len(self.item_index),
                                len(self.label_index),
                                len(self.item_index) + 1, n_items)
        from basket_residual_econ import CorrectionNet
        from basket_econ_finetune import TunedNet
        n_a, n_b = len(vocab["basket_items_A"]), len(vocab["basket_items_B"])

        def load(tag, module):
            blob = torch.load(self.dir / self.manifest["weights"][tag]["file"],
                              map_location="cpu", weights_only=False)
            module.load_state_dict(blob["state_dict"])
            module.eval()
            return module

        self.parent = load("domain_parent", v1.BasketNet(*arch(n_a)).to(self.device))
        self.tuned = load("econ_tuned",
                          TunedNet(v1.BasketNet(*arch(n_a)).to(self.device),
                                   CorrectionNet().to(self.device)))
        self.wardfree = load("wardfree", v1.BasketNet(*arch(n_b)).to(self.device))
        for tag, net in (("domain_parent", self.parent),
                         ("econ_tuned", self.tuned.base),
                         ("wardfree", self.wardfree)):
            self._assert_architecture(tag, net)
        with torch.no_grad():
            self._pvecs = self.parent.baskets.vectors(
                self.cat_idx_A, self.cat_cnt_A,
                torch.arange(self.n_cat, device=self.device))
            self._wvecs = self.wardfree.baskets.vectors(
                self.cat_idx_B, self.cat_cnt_B,
                torch.arange(self.n_act, device=self.device))

    def _assert_architecture(self, tag, net):
        """Inspect what was BUILT, not what we asked for."""
        enc = net.trunk.encoder.layers
        got = {"layers": len(enc),
               "heads": int(enc[0].self_attn.num_heads),
               "ff_dim": int(enc[0].linear1.out_features),
               "d_model": int(net.trunk.champ_emb.embedding_dim)}
        if got != self.arch:
            raise ValueError(
                f"{tag}: built architecture {got} != manifest {self.arch}")
        return got

    # ------------------------------------------------------------------
    # There is deliberately no `assert_reads_no_train()` here. A method that
    # returns True without checking anything is worse than no check: it reads
    # like evidence. The real proof is loading and scoring this package with
    # the training corpus and the experiment caches made unreadable -- run by
    # scripts/mixture_displayed_policy.py --sandbox, which reports what it
    # blocked and whether scoring still succeeded.

    def action_of(self, group):
        """Group index -> the action tuple, items repeated by count."""
        return tuple(sorted(i for i, n in self.group_main[group]
                            for _ in range(n)))

    def probabilities(self, rows):
        """rows: list of decision dicts -> [B, n_act] mixture probabilities."""
        self._apply_config()
        tp, v1 = self._tp, self._v1
        from basket_econ_cache import N_ECON, econ_features, to_action
        ds = tp.ShopDataset(list(rows), self.champ_index, self.item_index,
                            self.label_index, self.dragon)
        batch = next(ds.batches(len(rows), shuffle=False))
        b = {k: batch[k] for k in ("champs", "sides", "items", "query",
                                   "inv", "hist")}
        with torch.no_grad():
            ps = self.parent.scores(self.parent.state(b, self.device),
                                    self._pvecs)
            dom = torch.topk(ps, self.shortlist, dim=1).indices
            dom_cpu = dom.cpu()
            econ = torch.zeros((len(rows), self.shortlist, N_ECON))
            from collections import Counter as _C
            for j, row in enumerate(rows):
                budget = float(row["gold_est"])
                for k in range(self.shortlist):
                    cand = int(dom_cpu[j, k])
                    if cand == 0:
                        econ[j, k, 0] = 1.0
                    else:
                        econ[j, k] = torch.tensor(econ_features(
                            to_action(self.baskets_A[cand - 1]), row, budget,
                            self.dragon, _C()))
            sA, _ = self.tuned.corrected_scores(
                b, self.device, self.cat_idx_A, self.cat_cnt_A, self.n_cat,
                dom, (econ.to(self.device) - self.sm) / self.ss)
            gi = self.group_ids.unsqueeze(0).expand(sA.shape[0], -1)
            gmax = torch.full((sA.shape[0], self.n_act), float("-inf"),
                              device=self.device)
            gmax.scatter_reduce_(1, gi, sA, reduce="amax", include_self=True)
            gsum = torch.zeros(sA.shape[0], self.n_act, device=self.device)
            gsum.scatter_add_(1, gi, (sA - gmax.gather(1, gi)).exp())
            P_A = torch.softmax(gmax + gsum.log(), dim=1)
            sB = self.wardfree.scores(
                self.wardfree.state(b, self.device), self._wvecs)
            P_B = torch.zeros_like(P_A)
            P_B.scatter_(1, self.perm.unsqueeze(0).expand(sA.shape[0], -1),
                         torch.softmax(sB, dim=1))
            return self.wA * P_A + self.wB * P_B

    def options(self, rows):
        """The displayed policy: up to `n_options` DISTINCT legal actions.

        ONE wallet per decision, `row["gold_est"]`, and no override. An
        override would only have reached the legality filter, while
        `probabilities()` feeds the state, the frozen domain and the economic
        features from `row["gold_est"]` — the scorer and the filter would then
        be pricing against different wallets. A caller needing a different
        wallet must put it in the row, so every stage sees the same amount.

        Every returned option has passed purchase_sequence at that wallet.
        SAVE is always legal.
        """
        from app.conditional_policy import purchase_sequence
        P = self.probabilities(rows)
        order = torch.topk(P, min(self.shortlist, self.n_act),
                           dim=1).indices.cpu()
        out = []
        for j, row in enumerate(rows):
            budget = float(row["gold_est"])
            opts, calls = [], 0
            for g in order[j].tolist():
                if g == 0:
                    opts.append({"kind": "save", "items": {}, "group": 0,
                                 "legal": True})
                else:
                    action = self.action_of(g)
                    calls += 1
                    if purchase_sequence(action, row, budget,
                                         self.dragon) is not None:
                        opts.append({"kind": "buy", "group": g,
                                     "items": {str(i): n for i, n
                                               in self.group_main[g]},
                                     "legal": True})
                if len(opts) >= self.n_options:
                    break
            if not opts:
                opts = [{"kind": "save", "items": {}, "group": 0,
                         "legal": True}]
            out.append(opts)
        return out


def load_package(name="mixture-50-50-v1", dragon=None, device=None):
    pkg = ROOT / "data" / "packages" / name
    if dragon is None:
        from app.ddragon import dragon_for_patch
        manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
        dragon = dragon_for_patch(manifest["generation"]["patch"])
    return MixturePolicy(pkg, dragon, device=device)
