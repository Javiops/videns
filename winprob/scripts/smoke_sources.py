"""A bounded schema smoke on six admitted non-reserved games, no model metrics."""
from __future__ import annotations
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from winprob.admission import extraction_candidates, digest_bytes
from winprob.features import extract_game_states


def main():
    manifest_path = ROOT/'artifacts/admission/w1-admission-20260929-v1/admission_manifest.json'
    raw = manifest_path.read_bytes()
    expected = manifest_path.with_suffix('.sha256').read_text().split()[0]
    if digest_bytes(raw) != expected:
        raise ValueError('manifest digest mismatch')
    manifest = json.loads(raw)
    candidates = extraction_candidates(manifest, 6)
    archive = Path(manifest['source_data_root'])/'raw_match_v5'
    result = []
    for entry in candidates:
        mid = entry['match_id']
        payloads = {}
        for endpoint in ('match', 'timeline'):
            with gzip.open(archive/endpoint/f'{mid}.json.gz', 'rt', encoding='utf-8') as source:
                envelope = json.load(source)
            if (envelope['schema'] != 1 or envelope['match_id'] != mid
                    or envelope['endpoint'] != endpoint):
                raise ValueError('envelope mismatch')
            payloads[endpoint] = envelope['payload']
        if payloads['match']['info']['queueId'] != 420:
            raise ValueError('unexpected queue')
        rows = extract_game_states(**payloads)
        result.append({'match_id': mid, 'role': entry['role'], 'frames': len(rows),
                       'patch': entry['patch'], 'feature_count': len(rows[0]['x']) if rows else 0})
    report = {'kind': 'schema_smoke_not_performance', 'games': len(result),
              'test_payload_reads': False, 'confirmatory_payload_reads': False, 'results': result}
    dest = ROOT/'data/reports/source_smoke_v1.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
