"""Read a versioned producer artifact and verify explicit IQL accounting."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from winprob.trajectory_contract import SCHEMA, compile_transitions


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def bound_json(path):
    expected=path.with_suffix('.sha256').read_text(encoding='ascii').split()
    if expected!=[digest(path),path.name]:
        raise ValueError('Artifact sidecar mismatch')
    return json.loads(path.read_text(encoding='utf-8'))


def run(directory,output):
    manifest_path=directory/'manifest.json'
    manifest=bound_json(manifest_path)
    if manifest.get('schema')!=SCHEMA or manifest.get('state')!='complete':
        raise ValueError('Not a complete source-owned trajectory export')
    if any(manifest.get(key) is not False for key in ('test_payload_reads','confirmation_payload_reads','mixed_export_reads')):
        raise ValueError('Unverified source admission scope')
    rows_path=directory/'trajectories.jsonl'
    if digest(rows_path)!=manifest['rows_sha256'] or rows_path.with_suffix('.sha256').read_text().split()!=[manifest['rows_sha256'],rows_path.name]:
        raise ValueError('Trajectory payload digest mismatch')
    selected=set(manifest['selected_game_ids'])
    seen=set();counts=Counter();reasons=Counter();per_game=Counter()
    with rows_path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            key=row['trajectory_id']
            if key in seen or row['match_id'] not in selected or row['consumer_role']!='fit':
                raise ValueError('Unexpected or duplicate trajectory')
            seen.add(key);counts['player_games']+=1;per_game[row['match_id']]+=1
            if row['whole_player_game_eligible'] is not True:
                counts['excluded_player_games']+=1
                reasons.update(row['ineligibility_reasons'])
                continue
            transitions=compile_transitions(row)
            counts['eligible_player_games']+=1
            counts['transitions']+=len(transitions)
            counts['terminal_transitions']+=sum(t['terminated'] for t in transitions)
            counts['one_time_terminal_rewards']+=sum(t['reward'] for t in transitions)
            counts['last_decision_receives_terminal_reward']+=int(transitions[-1]['reward']==row['terminal']['reward'])
    if set(per_game)!=selected or any(n!=10 for n in per_game.values()):
        raise ValueError('Expected every participant in each selected game')
    if counts['eligible_player_games']!=manifest['eligible_player_games'] or len(seen)!=manifest['trajectory_count']:
        raise ValueError('Producer/consumer population differs')
    if counts['last_decision_receives_terminal_reward']!=counts['eligible_player_games']:
        raise ValueError('Terminal reward did not reach every final action')
    report={'schema':'winprob-trajectory-consumer-check-v1','status':'passed',
            'scope':'ten-game source-fit plumbing smoke, not policy performance',
            'source_manifest':str(manifest_path.resolve()),'manifest_sha256':digest(manifest_path),
            'rows_sha256':manifest['rows_sha256'],'games':len(selected),
            'counts':dict(counts),'exclusions':dict(reasons),
            'reward_convention':'arrival reward: transition reward equals next node reward; terminal bootstrap multiplier zero',
            'producer_runtime_imported':False,'raw_payloads_read':False,
            'policy_trained':False,'practical_benefit_established':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.artifact.resolve(),args.output.resolve()),indent=2))
