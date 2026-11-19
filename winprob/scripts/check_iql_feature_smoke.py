"""Validate a small source-owned FIT trajectory artifact without reopening raw data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boots_support import load_static
from winprob.full_action_catalogue import load_catalogue, encode_complete_episode
from winprob.iql_features import PublicFeatureCompiler
from winprob.preference_reference import file_hash


def run(directory, output):
    manifest_path = directory/'manifest.json'
    if file_hash(manifest_path) != (directory/'manifest.sha256').read_text().split()[0]:
        raise ValueError('trajectory manifest digest mismatch')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['schema'] != 'winprob-fit-trajectories-v2' or manifest['state'] != 'complete'
            or manifest['full_mode'] is not False or len(manifest['requested_game_ids']) > 10):
        raise ValueError('only complete bounded FIT smoke artifacts allowed')
    for key in ('calibration_game_payload_reads', 'development_game_payload_reads',
                'test_payload_reads', 'confirmation_payload_reads', 'policy_loaded', 'training_performed'):
        if manifest[key] is not False:
            raise ValueError('invalid source scope')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_hash(directory/name) != expected:
            raise ValueError('trajectory file binding mismatch')
    catalogue, lookup = load_catalogue(manifest['full_action_catalogue_path'])
    if file_hash(manifest['full_action_catalogue_path']) != manifest['full_action_catalogue_sha256']:
        raise ValueError('catalogue identity changed')
    static, _ = load_static(ROOT/'artifacts/boots-static/20260930-v1/boots_static.json')
    compiler = PublicFeatureCompiler(static)
    for basket, kind in zip(catalogue['full_actions'], catalogue['full_action_kinds'], strict=True):
        compiler.full_action(basket, kind)
    episodes = transitions = 0
    games = set(); raw_hash = hashlib.sha256()
    with gzip.open(directory/manifest['rows_file'], 'rb') as handle:
        for line in handle:
            raw_hash.update(line)
            record = json.loads(line)
            if record['match_id'] not in manifest['requested_game_ids'] or record['consumer_role'] != 'fit':
                raise ValueError('trajectory escaped declared FIT IDs')
            encoded = encode_complete_episode(record, lookup)
            if not encoded['eligible']:
                raise ValueError('producer exported an unsupported episode')
            for node, compiled in zip(record['decisions'], encoded['transitions'], strict=True):
                if node['action']['full_action_id'] != compiled['full_action_id']:
                    raise ValueError('independent full-action ID mismatch')
                compiler.state(node['prestate'], record['patch'])
                transitions += 1
            episodes += 1; games.add(record['match_id'])
    if (raw_hash.hexdigest() != manifest['uncompressed_rows_sha256']
            or episodes != manifest['trajectory_count'] or transitions != manifest['counts']['decision_nodes']):
        raise ValueError('smoke stream content/count mismatch')
    report = {'state': 'passed', 'source_manifest_sha256': file_hash(manifest_path),
              'games': len(games), 'episodes': episodes, 'transitions': transitions,
              'full_catalogue_actions_encoded': len(catalogue['full_actions']),
              'global_features': 37, 'inventory_features_per_player': 22,
              'raw_sources_reopened': False, 'training_performed': False}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.directory.resolve(strict=True), args.output.resolve())
