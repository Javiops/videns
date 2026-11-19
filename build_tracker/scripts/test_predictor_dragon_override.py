"""Dragon-override contract: cross-corpus evaluation governance on Predictor.

Asserted properties:
  1. Without an override, an artifact whose static-data digest does not match
     its dragon is refused (the existing guard is unchanged).
  2. With an explicit override, the digest equality is waived but the waiver
     is RECORDED (`static_override`) and the predictor is never
     serving-eligible.
  3. The footwear-regime consistency check is never waived by an override.
"""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import torch

import train_prefix as tp
from app.ddragon import DataDragon
from app.predictor import Predictor


def build_artifact(path: Path, config: dict) -> None:
    champ_index, item_index, label_ids = {1: 1}, {1001: 1}, [1001]
    # Mirror Predictor's constructor exactly so only the properties under test
    # can differ between saving and loading.
    tp.apply_config(config)
    model = tp.PrefixModel(len(champ_index), len(item_index), len(label_ids),
                           len(item_index) + 1,
                           d_model=int(config.get('d_model', 64)),
                           layers=int(config.get('layers', 2)),
                           ff_dim=int(config.get('ff_dim', 128)),
                           heads=int(config.get('heads', 4)),
                           query_dim=tp.QUERY_DIM)
    torch.save({
        'config': config,
        'state_dict': model.state_dict(),
        'champ_index': champ_index,
        'item_index': item_index,
        'label_ids': label_ids,
        'provenance': {'status': 'declared', 'gold_input': 'prequential-v3',
                       'scored_on': 'validation'},
        'trained_at': '2026-09-14T00:00:00',
    }, path)


def main() -> None:
    dragon = DataDragon(None)
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / 'override_test.pt'
        config = {'query_dim': tp.BASE_QUERY_DIM, 'extras': True,
                  'static_data_sha256': 'not-the-governing-signature'}
        build_artifact(artifact, config)

        # 1. Default path still refuses a static-data digest mismatch.
        try:
            Predictor(artifact)
        except ValueError as exc:
            assert 'static-data digest mismatch' in str(exc), exc
        else:
            raise AssertionError('Digest mismatch was accepted without an override')

        # 2. The override waives equality but records the waiver and is never
        #    serving-eligible.
        predictor = Predictor(artifact, dragon_override=dragon)
        assert predictor.static_override is not None
        assert predictor.static_override['governing_signature'] == dragon.signature
        assert predictor.static_override['artifact_static_data_sha256'] == 'not-the-governing-signature'
        assert predictor.dragon is dragon
        assert predictor.deployment.get('eligible') is False
        assert 'not a serving configuration' in (predictor.deployment.get('reason') or '')

        # 3. Footwear regime consistency is never waived by an override.
        artifact2 = Path(tmp) / 'override_footwear.pt'
        config2 = {'query_dim': tp.BASE_QUERY_DIM, 'extras': True,
                   'footwear_state_version': 'footwear-timer-slots-v1'}
        build_artifact(artifact2, config2)
        try:
            Predictor(artifact2, dragon_override=dragon)
        except ValueError as exc:
            assert 'Footwear artifact' in str(exc), exc
        else:
            raise AssertionError('Footwear inconsistency was accepted under an override')

    print('dragon-override contract holds: refusal unchanged, waiver recorded, footwear never waived')


if __name__ == '__main__':
    main()
