"""Produce the exact disjoint recovery list from the completed source audit."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.dataset_artifacts import sha256,write_json


def main():
    folder=ROOT/'data/acceptance/footwear_repair_20260911'
    path=folder/'timer_full_source_parallel.json'
    out=folder/'recovery_summary.json'
    if out.exists(): raise ValueError('Recovery summary already exists')
    report=json.loads(path.read_text(encoding='utf-8'))
    prior_path=ROOT/'data/acceptance/16.17_development_20260911/qualification.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    recovered=[r for r in report['games'] if r['ordinary_replay_explains_game']]
    ids={r['match_id'] for r in recovered}
    prior_ids=set(prior['accepted_match_ids'])
    if ids & prior_ids: raise ValueError('Footwear recovery overlaps prior no-footwear corpus')
    if len(ids)!=len(recovered): raise ValueError('Duplicate recovered game')
    guards=Counter()
    for row in recovered:
        guards.update(row['retained_rows_by_guard_ms'])
    statuses=Counter(row['status'] for row in report['games'])
    source_qa=Counter()
    for row in report['games']:
        qa=row.get('independent_qa')
        if qa is not None:
            source_qa['checked_games']+=1
            source_qa['passing_games']+=qa['passed']
    summary={'schema':1,'status':'recovery_measurement_complete','source_report':str(path),'source_report_sha256':sha256(path),
             'previous_qualification_sha256':sha256(prior_path),'checked_footwear_games':len(report['games']),
             'boot_specific_qa':dict(source_qa),'replay_status_counts':dict(statuses),
             'additional_source_consistent_games':len(recovered),
             'additional_games_by_split':dict(Counter(r['split'] for r in recovered)),
             'additional_candidate_rows_by_split':{s:sum(r['candidate_rows'] for r in recovered if r['split']==s) for s in ('train','val')},
             'candidate_rows_by_guard_ms':dict(guards),'previous_no_footwear_games':len(prior_ids),
             'prospective_combined_games_before_new_admission':len(prior_ids)+len(recovered),
             'recovered_match_ids':sorted(ids),'exclusion_and_failure_counts':report['exclusion_and_failure_counts'],
             'recovered_games':[{k:r[k] for k in ('match_id','split','candidate_rows','grant_count','full_slot_delayed_grants')} for r in recovered],
             'training_admission_performed':False,'final_test_payload_opened':False,
             'limits':report['limits']+['The previous and recovered game sets are disjoint. Their sum is prospective coverage, not an already-published dataset.',
                                      'Full-source source/outcome checks ran for every replay; causal prefix perturbations were performed on the fixed sample and synthetic tests.']}
    write_json(out,summary)
    compact={k:v for k,v in summary.items() if k not in ('recovered_match_ids','recovered_games','limits','exclusion_and_failure_counts')}
    print(json.dumps(compact,indent=2))


if __name__=='__main__': main()
