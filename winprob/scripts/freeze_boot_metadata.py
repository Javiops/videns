"""Freeze cached Data Dragon metadata for the boots feasibility experiment.

Reads six static JSON files only. No game, outcome, timeline, database or
network access is performed; original item objects are preserved in full.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ('16.17.1', '16.18.1', '16.19.1')
BOOTS = {'3047': ('Plated Steelcaps', 1200), '3111': ("Mercury's Treads", 1250)}


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_static(cache, kind, version):
    singular = {'items': 'item', 'champions': 'champion'}[kind]
    path = (cache / f'{singular}-{version}.json').resolve(strict=True)
    if not path.is_relative_to(cache):
        raise ValueError('Static source escaped the supplied cache directory')
    raw = path.read_bytes()
    payload = json.loads(raw.decode('utf-8-sig'))
    if payload.get('version') != version or not isinstance(payload.get('data'), dict) or not payload['data']:
        raise ValueError('Invalid static version or data mapping: ' + str(path))
    return payload['data'], {'path': str(path), 'sha256': sha(raw), 'kind': kind, 'bytes': len(raw)}


def export_metadata(cache_dir, output_root, export_version):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', export_version) is None:
        raise ValueError('Unsafe export version')
    cache = Path(cache_dir).resolve(strict=True)
    destination = Path(output_root).resolve() / export_version
    if destination.exists():
        raise FileExistsError('Static export version already exists: ' + str(destination))
    patches = {}
    checks = {}
    for version in VERSIONS:
        patch = '.'.join(version.split('.')[:2])
        items, item_source = read_static(cache, 'items', version)
        champions, champion_source = read_static(cache, 'champions', version)
        if not all(isinstance(item_id, str) and item_id.isdecimal() and isinstance(item, dict)
                   for item_id, item in items.items()):
            raise ValueError('Invalid original item mapping: ' + version)
        tags = {}
        for champion in champions.values():
            key = champion.get('key')
            values = champion.get('tags')
            if (not isinstance(key, str) or not key.isdecimal() or key in tags
                    or not isinstance(values, list) or not all(isinstance(tag, str) for tag in values)):
                raise ValueError('Invalid numeric champion key/tags: ' + version)
            tags[key] = values
        boots = {}
        for item_id, (expected_name, expected_total) in BOOTS.items():
            item = items.get(item_id, {})
            gold = item.get('gold', {})
            if (item.get('name') != expected_name or type(gold.get('total')) is not int
                    or gold['total'] != expected_total or gold.get('purchasable') is not True
                    or item.get('maps', {}).get('11') is not True):
                raise ValueError('Boot name/total/map/purchasability differs: ' + version + '/' + item_id)
            boots[item_id] = {'name': item['name'], 'total_gold': gold['total'],
                              'map_11': True, 'purchasable': True,
                              'full_original_object_sha256': sha(canonical(item))}
        patches[patch] = {'version': version, 'items': items, 'champion_tags': tags,
                          'source_files': [item_source, champion_source]}
        checks[patch] = {'item_count': len(items), 'champion_tags_count': len(tags), 'boots': boots}
    patch_names = tuple(patches)
    first = patches[patch_names[0]]
    equivalence = {item_id: all(patches[patch]['items'][item_id] == first['items'][item_id]
                               for patch in patch_names[1:]) for item_id in BOOTS}
    payload = {
        'schema': 'winprob-boots-static-v1', 'export_version': export_version,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_cache_directory': str(cache), 'patches': patches,
        'verification': {'patches': checks, 'boots_full_original_objects_equal_across_patches': equivalence,
                         'all_item_data_equal_across_patches': all(patches[patch]['items'] == first['items']
                                                                for patch in patch_names[1:]),
                         'champion_tags_equal_across_patches': all(patches[patch]['champion_tags'] == first['champion_tags']
                                                                 for patch in patch_names[1:])},
        'scope': 'Static metadata only; all original item objects retained for declared spend upper bounds.',
        'limitations': ['Equal cached boot objects do not establish equal in-game effects across patches.',
                        'Champion tags are broad Data Dragon categories, not measured damage mix or causal effect modifiers.'],
        'game_payload_reads': False, 'outcome_reads': False, 'network_reads': False,
        'exporter': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
    }
    for patch in patches.values():
        for source in patch['source_files']:
            if sha(Path(source['path']).read_bytes()) != source['sha256']:
                raise ValueError('Static source changed during export')
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'boots_static.json'
    raw = canonical(payload)
    with path.open('xb') as handle:
        handle.write(raw)
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(raw) + '  boots_static.json\n')
    return path, sha(raw), payload['verification']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT.parent/'build_tracker/data/cache')
    parser.add_argument('--output-root', type=Path, default=ROOT/'artifacts/boots-static')
    parser.add_argument('--export-version', default='20260930-v1')
    args = parser.parse_args()
    path, digest, verification = export_metadata(args.cache_dir, args.output_root, args.export_version)
    print(json.dumps({'path': str(path), 'sha256': digest, 'verification': verification,
                      'game_payload_reads': False}, indent=2))


if __name__ == '__main__':
    main()
