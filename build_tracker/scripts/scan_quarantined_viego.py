"""Bounded source inventory of quarantined Viego games; no model or test reads."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.dataset_artifacts import canonical_sha, sha256, write_json
from app.riot import RawMatchArchive


def scan(qualification, out):
    if out.exists():
        raise ValueError('Evidence exists; refusing overwrite')
    q = json.loads(qualification.read_text(encoding='utf-8'))
    corpus = Path(q['corpus'])
    manifest_path = corpus / 'corpus_manifest.json'
    if sha256(manifest_path) != q['corpus_manifest_sha256']:
        raise ValueError('Qualification/corpus binding changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source_index = corpus / 'source_inventory.jsonl'
    if sha256(source_index) != manifest['source_inventory_sha256']:
        raise ValueError('Frozen source inventory changed')
    sources = {r['match_id']:r for r in map(json.loads, source_index.read_text().splitlines())}
    selected = {mid:reasons for mid,reasons in q['quarantined_match_ids'].items()
                if 'unmodeled_viego_inventory' in reasons}
    if len(selected) > 150:
        raise ValueError('This investigation is bounded to at most 150 games')
    rows = []
    archive = RawMatchArchive(corpus / 'raw')
    for index, (mid,reasons) in enumerate(sorted(selected.items()), 1):
        payloads = {}
        for kind in ('match','timeline'):
            binding = sources[mid]['files'][kind]
            path = corpus / binding['path']
            if sha256(path) != binding['sha256']:
                raise ValueError(f'Raw file changed: {mid}/{kind}')
            payload = archive.read(kind,mid)
            if canonical_sha(payload) != binding['payload_sha256']:
                raise ValueError(f'Raw payload changed: {mid}/{kind}')
            payloads[kind] = payload
        match, timeline = payloads['match'],payloads['timeline']
        players = [p for p in match['info']['participants'] if p['championId'] == 234]
        if len(players) != 1:
            raise ValueError(f'Expected one Viego in ranked sample: {mid}')
        player = players[0]
        pid = player['participantId']
        raw_events = [e for f in timeline['info']['frames'] for e in f.get('events',[])]
        takedowns = [e for e in raw_events if e['type'] == 'CHAMPION_KILL'
                     and (e.get('killerId') == pid or pid in e.get('assistingParticipantIds',[]))]
        item_events = [e for e in raw_events if e.get('participantId') == pid and e['type'].startswith('ITEM_')]
        batches = {}
        for e in item_events:
            batches.setdefault(e['timestamp'], []).append(e)
        destroy_only = [{'ts':ts,'items':[e['itemId'] for e in batch]}
                        for ts,batch in batches.items()
                        if all(e['type'] == 'ITEM_DESTROYED' for e in batch)]
        rows.append({'match_id':mid,'participant_id':pid,'duration_s':match['info']['gameDuration'],
                     'original_reasons':reasons,'takedown_count':len(takedowns),
                     'first_takedown_ts':min((e['timestamp'] for e in takedowns),default=None),
                     'item_event_counts':dict(Counter(e['type'] for e in item_events)),
                     'standalone_destroy_batches':destroy_only,
                     'raw_sources_sha256':{k:sources[mid]['files'][k]['sha256'] for k in ('match','timeline')}})
        if index % 25 == 0:
            print(f'Scanned {index}/{len(selected)} quarantined Viego sources', flush=True)
    no_takedowns = [r['match_id'] for r in rows if r['takedown_count'] == 0]
    report = {'schema':1,'purpose':'quarantined_source_screening_only','created_utc':datetime.now(timezone.utc).isoformat(),
              'qualification':str(qualification),'qualification_sha256':sha256(qualification),
              'corpus_manifest_sha256':sha256(manifest_path),'code_sha256':sha256(Path(__file__)),
              'games_scanned':len(rows),'no_takedown_match_ids':no_takedowns,
              'duration_eligible_no_takedown_match_ids':[r['match_id'] for r in rows if r['takedown_count'] == 0 and r['duration_s'] >= 600],
              'games':rows,'training_clearance':False,'final_test_payload_opened':False,
              'limits':['No-takedown screening identifies candidates only; all reconstruction and inventory checks must still pass.',
                        'A takedown permits possession but does not prove it happened.',
                        'A matching terminal inventory alone does not validate intermediate states.']}
    write_json(out,report)
    print(json.dumps({k:report[k] for k in ('games_scanned','no_takedown_match_ids','duration_eligible_no_takedown_match_ids')},indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    scan(a.qualification.resolve(),a.out.resolve())
