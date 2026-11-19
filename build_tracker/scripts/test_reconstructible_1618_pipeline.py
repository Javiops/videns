"""Two-game real-source build/audit smoke; no reserved-test payloads are read."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from process_reconstructible_1618 import BASE,build,audit_and_qualify,game_checks
from reconstructible_1618_runtime import activate
from app.dataset_artifacts import write_json

plan=json.loads((BASE/'source_plan.json').read_text(encoding='utf-8'))
activate(plan['static_review'])
from app.ddragon import dragon_for_patch
from app.reconstruct import reconstruct_game
from app.riot import RawMatchArchive
dragon=dragon_for_patch('16.18');archive=RawMatchArchive(ROOT/'data/raw_match_v5')
records={r['match_id']:r for r in plan['matches']}
chosen=['KR_8375431152']
assert chosen[0] in records and chosen[0] not in plan['reserved_test_ids']
for mid in list(records)[:12]:
    match=archive.read('match',mid)
    if match['info']['gameDuration']<600 or any(p['championId']==234 for p in match['info']['participants']): continue
    timeline=archive.read('timeline',mid);debug={}
    _,events=reconstruct_game(match,timeline,dragon,debug)
    reasons,checks=game_checks(match,timeline,events,debug,dragon)
    if not reasons and debug.get('footwear_grants'):
        chosen.append(mid);break
assert len(chosen)==2,'No bounded positive footwear source found'
with tempfile.TemporaryDirectory(prefix='reconstructible1618-smoke-',dir=ROOT/'data') as temporary:
    folder=Path(temporary)
    assert folder.resolve().is_relative_to((ROOT/'data').resolve())
    scoped={**plan,'matches':[records[mid] for mid in chosen],'selected_sources':2,'test_fixture':True}
    source=folder/'source_plan.json';write_json(source,scoped)
    corpus=folder/'corpus'
    build(source,corpus)
    screening=json.loads((corpus/'screening_report.json').read_text(encoding='utf-8'))
    assert screening['accepted_games']==2 and screening['excluded_games']==0,screening['exclusion_reason_counts']
    audit_and_qualify(source,corpus,folder/'evidence/semantic_audit.json')
    qualification=json.loads((folder/'evidence/qualification.json').read_text(encoding='utf-8'))
    assert qualification['accepted_games']==2 and qualification['training_clearance'] is False
    assert qualification['test_payload_reads'] is False and any(k.startswith('viego') for k in qualification['accepted_cohorts'])
print('ok real16.18 pipeline: strict Viego + footwear, frozen source masks, independent full semantic audit, qualified isolated DB; no test or export')
