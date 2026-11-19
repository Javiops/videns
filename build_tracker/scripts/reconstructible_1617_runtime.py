"""The reviewed 16.17 reconstruction, loaded through the SAME isolated mechanism as the 16.18 variant.

Derived from reconstructible_1618_runtime.py (2026-09-27, prior-patch transfer, owner GO). The
reviewed sources are loaded VERBATIM (their 16.17 guards are asserted, nothing is substituted)
so build, audit and export bind the exact 16.17 semantics with the same version bookkeeping;
source hashes are recorded and the pinned 16.17.1 static data is checked before activation.
This module does not enable training or serving.
"""
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256

RECONSTRUCTION_VERSION='decision-start-footwear-v4'      # the reviewed base label, unchanged
INVENTORY_VERSION='footwear-timer-inventory-elixirs-prototype-v1'   # the reviewed base label, unchanged
STATE_VERSION='footwear-timer-slots-v1'
SOURCES={'footwear_economics':'scripts/footwear_economics.py',
         'footwear_timer_inventory':'scripts/footwear_timer_inventory.py',
         '_reconstructible_1617_replay':'app/footwear_reconstruction.py'}
_ACTIVE=False


def variants():
    result={}
    for name,path in SOURCES.items():
        source=(ROOT/path).read_text(encoding='utf-8')
        if '16.17' not in source: raise ValueError('Reviewed patch guard is missing: '+path)
        changed=source                                   # reviewed 16.17 sources, verbatim
        if name=='footwear_timer_inventory' and changed.count(INVENTORY_VERSION)!=1: raise ValueError('Inventory version drift')
        if name=='_reconstructible_1617_replay' and changed.count(RECONSTRUCTION_VERSION)!=1: raise ValueError('Reconstruction version drift')
        result[name]=changed
    return result


def declaration():
    return {'schema':1,'patch':'16.17','runtime_version':'reconstructible-16.17-v1',
        'reconstruction_version':RECONSTRUCTION_VERSION,'inventory_version':INVENTORY_VERSION,
        'footwear_state_version':STATE_VERSION,'economics_version':'footwear-economics-16.17-v1',
        'timing_guard_ms':1000,'delivery_time_independently_observed':False,
        'base_code_sha256':{path:sha256(ROOT/path) for path in SOURCES.values()},
        'variant_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in variants().items()}}


def validate_state(row):
    if (row.get('footwear_state_version')!=STATE_VERSION
        or type(row.get('magical_footwear_selected')) is not bool
        or type(row.get('footwear_granted')) is not bool
        or row['footwear_granted'] and not row['magical_footwear_selected']):
        raise ValueError('Missing or inconsistent footwear input state')
    return {k:row[k] for k in ('footwear_state_version','magical_footwear_selected','footwear_granted')}


def activate(review_path):
    global _ACTIVE
    if _ACTIVE: return declaration()
    forbidden={'footwear_economics','footwear_timer_inventory','baseline','audit_patch_corpus',
               'app.footwear_reconstruction','train_prefix','app.predictor'}
    if forbidden & sys.modules.keys(): raise RuntimeError('Activate 16.17 before importing reconstruction consumers')
    review_path=Path(review_path)
    review=json.loads(review_path.read_text(encoding='utf-8'))
    if (review.get('patch')!='16.17' or review.get('ddragon_version')!='16.17.1'
        or review.get('timer_economics_and_physical_rules_compatible') is not True):
        raise ValueError('16.17 mechanics have not passed the static review')
    for name,digest in review['files'].items():
        if sha256(review_path.parent/name)!=digest: raise ValueError('Static evidence changed')
    if sha256(ROOT/'data/cache/item-16.17.1.json')!=review['files']['item-16.17.1.json']:
        raise ValueError('Active pinned item data differs from reviewed source')
    import app.ddragon as ddragon
    import app.shop_econ as economics
    import app.reconstruct as reconstruction
    original_dragon=ddragon.dragon_for_patch('16.17')
    if original_dragon.version!='16.17.1': raise ValueError('Wrong pinned patch data')
    loaded={}
    for name,text in variants().items():
        module=types.ModuleType(name)
        module.__file__=str(ROOT/SOURCES[name])
        sys.modules[name]=module
        exec(compile(text,module.__file__,'exec'),module.__dict__)
        loaded[name]=module
    adapted=loaded['footwear_economics']
    dragon=adapted.FootwearDataDragon(original_dragon)
    dragon.footwear_state_version=STATE_VERSION
    def pinned(patch):
        if patch!='16.17': raise ValueError('This isolated runtime is patch16.17 only')
        return dragon
    ddragon.dragon_for_patch=pinned
    ddragon.default_dragon=lambda:dragon
    for name in ('combine_cost','components_of','is_blocked','try_purchase','buy_cost','inventory_state','decision_kind','net_spent'):
        setattr(economics,name,getattr(adapted,name))
    replay=loaded['_reconstructible_1617_replay']
    reconstruction.reconstruct_game=replay.reconstruct_game
    reconstruction.RECONSTRUCTION_VERSION=RECONSTRUCTION_VERSION
    reconstruction.dragon_for_patch=pinned
    import baseline
    old_example,old_others=baseline._example,baseline._others
    def others(event):
        rows=old_others(event)
        members=[p for p in event.get('board',[]) if not p.get('is_self')]
        if len(rows)!=len(members): raise ValueError('Board projection changed')
        for row,member in zip(rows,members,strict=True): row.update(validate_state(member))
        return rows
    def example(event,data):
        if event.get('reconstruction_version')!=RECONSTRUCTION_VERSION: raise ValueError('Wrong replay version')
        state=validate_state(event)
        for member in event.get('board',[]): validate_state(member)
        row=old_example(event,data)
        if row is not None: row.update(state)
        return row
    baseline._others=others
    baseline._example=example
    _ACTIVE=True
    return declaration()
