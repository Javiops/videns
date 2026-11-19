"""Verify a slot-state rebuild without opening final-test payloads.

Before publication, require the initial benchmark's exact split membership.
After publication, verify generation bindings and stream only train/validation
rows through the explicit slot-state schema. This is data QA, not evaluation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json
from app.dataset_generations import verify_generation
from publish_patch_dataset import validate_slot_state


def check_assignment(assignment, prior):
    if assignment != prior:
        changes = Counter(f'{prior.get(mid, "absent")}->{assignment.get(mid, "absent")}'
                          for mid in prior.keys() | assignment.keys()
                          if prior.get(mid) != assignment.get(mid))
        raise ValueError(f'Initial rebuild changed frozen split membership: {dict(changes)}')


def verify(*, coverage=None, generation=None, prior_generation, out):
    if out.exists():
        raise ValueError('Verification evidence already exists')
    prior_path = prior_generation / 'split_assignment.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if coverage is not None:
        plan_path = coverage / 'split_plan.json'
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        assignment = plan['assignment']
        report = {'purpose':'prepublication_split_preservation', 'split_plan_sha256':sha256(plan_path)}
    else:
        manifest = verify_generation(generation)
        assignment = json.loads((generation / 'split_assignment.json').read_text(encoding='utf-8'))
        report = {'purpose':'export_schema_verification', 'generation':str(generation),
                  'generation_manifest_sha256':sha256(generation / 'generation_manifest.json'),
                  'export_fingerprint':manifest['export_fingerprint']}
    check_assignment(assignment, prior)
    report.update(prior_generation=str(prior_generation),
                  prior_split_assignment_sha256=sha256(prior_path),
                  identical_split_membership=True, games_by_split=dict(Counter(assignment.values())))
    if generation is not None:
        rows, games, completions = Counter(), {}, Counter()
        for split in ('train', 'val'):
            seen = set()
            with (generation / f'visits_{split}.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    validate_slot_state(row)
                    if assignment.get(row['match_id']) != split:
                        raise ValueError('Exported row disagrees with frozen assignment')
                    rows[split] += 1
                    seen.add(row['match_id'])
                    completions[f'{split}:{row["bot_quest_complete"]}'] += 1
            if seen != {mid for mid, assigned in assignment.items() if assigned == split}:
                raise ValueError('Exported games disagree with frozen assignment')
            if rows[split] != manifest['files'][f'visits_{split}.jsonl']['rows']:
                raise ValueError('Exported row count disagrees with manifest')
            games[split] = len(seen)
        report.update(checked_rows_by_split=dict(rows), checked_games_by_split=games,
                      bot_quest_complete_rows=dict(completions), slot_state_version='role-slots-v1')
    report.update(passed=True, final_test_payload_opened=False, model_evaluation_performed=False)
    write_json(out, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--coverage', type=Path)
    mode.add_argument('--generation', type=Path)
    p.add_argument('--prior-generation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verify(coverage=a.coverage.resolve() if a.coverage else None,
           generation=a.generation.resolve() if a.generation else None,
           prior_generation=a.prior_generation.resolve(), out=a.out.resolve())
