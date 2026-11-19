"""Publish frozen full retained-basket action IDs using package metadata only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'data/packages/mixture-v4-m2-v1'
DEFAULT_OUTPUT=ROOT/'data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1'
SCHEMA='winprob-full-action-catalogue-v1'
SAVE_SENTINEL=999999
WARDS={2055,772043}


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def parse_basket(key):
    require(isinstance(key,str) and key,'Expected nonempty full-basket catalogue key')
    items=[]
    for part in key.split(','):
        pieces=part.split(':');require(len(pieces)==2,'Malformed basket item/count')
        item,count=pieces
        require(item.isdecimal() and count.isdecimal() and str(int(item))==item and str(int(count))==count,
                'Noncanonical basket item/count')
        item,count=int(item),int(count)
        require(item>0 and item!=SAVE_SENTINEL and count>0,'Invalid physical basket item/count')
        items.append([item,count])
    require(items==sorted(items) and len({item for item,count in items})==len(items),'Basket items unsorted/duplicated')
    return items


def construct(catalogues):
    full=[[]]+[parse_basket(key) for key in catalogues['baskets_A']]
    groups=[[[int(item),int(count)] for item,count in group] for group in catalogues['group_main']]
    require(groups and groups[0]==[],'Main group zero must be empty')
    group_index={canonical(group):index for index,group in enumerate(groups)}
    require(len(group_index)==len(groups) and len({canonical(action) for action in full})==len(full),'Duplicate canonical action/group')
    mapping=[];variants=Counter();zero_ward=set()
    for action in full:
        main=[entry for entry in action if entry[0] not in WARDS]
        require(canonical(main) in group_index,'Full action has no matching frozen main group')
        group=group_index[canonical(main)];mapping.append(group);variants[group]+=1
        if main==action:zero_ward.add(group)
    require(set(variants)==set(range(len(groups))),'A main group lacks a full-A variant')
    return {'full_actions':full,'full_action_kinds':['observed_no_buy_death']+['retained_net_purchase']*(len(full)-1),
            'full_to_group':mapping,'group_main':groups,'full_action_count':len(full),'group_count':len(groups),
            'groups_with_zero_ward_variant':len(zero_ward),'groups_without_zero_ward_variant':len(groups)-len(zero_ward),
            'group0_full_action_ids':[index for index,group in enumerate(mapping) if group==0],
            'variant_count_distribution':dict(sorted(Counter(variants.values()).items()))}


def export(output=DEFAULT_OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Immutable full-action catalogue already exists')
    manifest_path=PACKAGE/'manifest.json';identity_path=PACKAGE/'package_id.json';catalogues_path=PACKAGE/'catalogues.json'
    manifest=json.loads(manifest_path.read_text());identity=json.loads(identity_path.read_text())
    require(hashlib.sha256(canonical(manifest)).hexdigest()==identity['manifest_sha256'],'Frozen package identity changed')
    require(digest(catalogues_path)==manifest['catalogues']['sha256'],'Frozen full/main catalogue digest changed')
    require(set(manifest['policy']['wards_stripped'])==WARDS,'Frozen ward mapping changed')
    data=construct(json.loads(catalogues_path.read_text()))
    require(data['full_action_count']==13166 and data['group_count']==9333,'Unexpected frozen catalogue dimensions')
    source_files={str(path):digest(path) for path in (manifest_path,identity_path,catalogues_path)}
    static_files={str(ROOT/f'data/cache/{kind}-16.18.1.json'):digest(ROOT/f'data/cache/{kind}-16.18.1.json')
                  for kind in ('item','champion')}
    code_files={str(ROOT/name):digest(ROOT/name) for name in ('scripts/export_winprob_action_catalogue_v1.py',
                'app/mixture_policy.py','app/purchase_multiset.py','scripts/basket_vocabulary_probe.py','scripts/basket_wardfree.py')}
    payload={'schema':SCHEMA,'version':output.name,'created_utc':datetime.now(timezone.utc).isoformat(),**data,
             'save_action_id':0,'source_save_sentinel':SAVE_SENTINEL,'source_save_label_counts':{str(SAVE_SENTINEL):1},
             'wards_stripped_only_for_group_projection':sorted(WARDS),'action_encoding':'canonical sorted physical item/count multiset with explicit empty no-buy tag',
             'source_action_contract':'Complete retained purchase targets, not primitive transaction sequences; wards and copy counts are retained.',
             'package':'mixture-v4-m2-v1','package_identity_sha256':identity['manifest_sha256'],
             'ddragon_version':'16.18.1','static_data_sha256':manifest['static_data_sha256'],
             'source_files_sha256':source_files,'static_files_sha256':static_files,'code_sha256':code_files,
             'game_payload_reads':False,'policy_loaded':False,'training_performed':False,
             'limits':['Catalogue support is not proof of conditional support in every state.',
                       'An absent full action must never be mapped to a ward-stripped substitute.',
                       'SAVE is an explicit observed no-buy-death target, not a physical item or a label for unobserved manual waits.']}
    output.mkdir(parents=True,exist_ok=False)
    path=output/'catalogue.json'
    path.write_bytes(canonical(payload)+b'\n')
    path.with_suffix('.sha256').write_text(digest(path)+'  catalogue.json\n',encoding='ascii')
    return path,payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();path,payload=export(args.output)
    print(json.dumps({'path':str(path),'sha256':digest(path),'full_action_count':payload['full_action_count'],
                      'group_count':payload['group_count'],'group0_full_action_ids':payload['group0_full_action_ids'],
                      'game_payload_reads':False,'policy_loaded':False},indent=2))


if __name__=='__main__':main()
