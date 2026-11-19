"""Strict independent Viego ownership ledger for a scoped 16.19 source audit.

No possession interval is guessed and no last-known inventory is substituted.
Unexplained destruction, borrowed items, unsupported role transforms and undo
ambiguity reject the game. Outcome inventories are consulted only after replay.
"""
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict

VERSION='viego-strict-source-16.19-v1'


def ids(payload):
    result=[]
    for item in payload or []:
        if isinstance(item,dict):
            result.extend([int(item['item_id'])]*int(item.get('count') or 1))
        else: result.append(int(item))
    return result


def recipe_quote(item,inventory,dragon):
    available=inventory.copy();used=Counter()
    magical_parents={int(x) for x in (dragon.item(2422) or {}).get('into',[])}
    def acquire(current,root=False,parent=None,ancestors=()):
        if current in ancestors: raise ValueError('Cyclic item recipe')
        physical=current
        if not root and not available[current] and current==1001 and available[2422] and parent in magical_parents:
            physical=2422
        if not root and available[physical]:
            available[physical]-=1;used[physical]+=1;return 0
        data=dragon.item(current)
        if not data: raise ValueError('Unknown recipe item')
        children=[int(x) for x in data.get('from',[])]
        return (int(data['gold']['total']) if not children else int(data['gold']['base'])+
            sum(acquire(child,parent=current,ancestors=ancestors+(current,)) for child in children))
    return acquire(item,root=True),used


def audit_viego_source(match,timeline,dragon,*,replay_events=None,debug=None,footwear_qa=None):
    if dragon.version!='16.19.1': raise ValueError('Viego gate requires pinned16.19.1')
    from footwear_timer import physical_slot_state
    from validate_footwear_timer_results import grant_boundary_reason
    players=match['info']['participants']
    owners=[p for p in players if p['championId']==234]
    if len(owners)!=1: raise ValueError('Expected exactly one Viego')
    actor=owners[0];owner=actor['participantId'];role=actor.get('teamPosition') or actor.get('individualPosition')
    failures=[];ledger=[];rules=Counter();inventory=Counter();purchases=defaultdict(list);sales=defaultdict(list)
    def fail(stamp,code,**detail): failures.append({'timestamp':stamp,'code':code,**detail})
    if role not in ('TOP','JUNGLE'): fail(0,'unsupported_implicit_role_transform',role=role)
    # Frame championStats are not used by the model. Its level/CS/gold inputs
    # must still be consistent with the owner's persistent progression.
    previous={}
    for frame in timeline['info']['frames']:
        data=frame.get('participantFrames',{}).get(str(owner),{})
        for field in ('level','xp','minionsKilled','jungleMinionsKilled','totalGold'):
            value=data.get(field)
            if value is not None:
                if not isinstance(value,(int,float)) or value<previous.get(field,0):
                    fail(frame['timestamp'],'nonmonotone_owner_progression',field=field)
                previous[field]=value
    batches=defaultdict(list)
    for fi,frame in enumerate(timeline['info']['frames']):
        for ei,event in enumerate(frame.get('events',[])):
            if event.get('participantId')==owner and event.get('type','').startswith('ITEM_'):
                batches[event['timestamp']].append((fi,ei,event))
            elif event.get('creatorId')==owner and event.get('type')=='WARD_PLACED':
                batches[event['timestamp']].append((fi,ei,event))
    all_grants=(debug or {}).get('footwear_grants',[])
    grants={g['timestamp']:g for g in all_grants if g['participant_id']==owner}
    if len(grants)!=sum(g['participant_id']==owner for g in all_grants): fail(0,'duplicate_footwear_grant')
    selected=any(r.get('perk')==8304 for s in actor.get('perks',{}).get('styles',[]) for r in s.get('selections',[]))
    if selected and (footwear_qa or {}).get('passed') is not True: fail(0,'missing_independent_footwear_qa')
    if grants and not selected: fail(0,'footwear_grant_without_rune')
    transformations={}
    for raw_id,data in dragon._items['data'].items():
        source=data.get('specialRecipe')
        if source and data.get('maps',{}).get('11') and data.get('gold',{}).get('purchasable') is False:
            source=int(source)
            if source in transformations: fail(0,'ambiguous_static_transformation',item_id=source)
            transformations[source]=int(raw_id)
    quest_seen=set();states_before={};states_after={};boundary_states={}
    def bag(): return sorted((+inventory).elements())
    for stamp in sorted(set(batches)|set(grants)):
        entries=sorted(batches[stamp],key=lambda x:x[:2]);states_before[stamp]=bag();boundary_states[stamp]=[bag()]
        bought=[e for _,_,e in entries if e['type']=='ITEM_PURCHASED']
        destroyed=Counter(e.get('itemId') for _,_,e in entries if e['type']=='ITEM_DESTROYED')
        wards=[e for _,_,e in entries if e['type']=='WARD_PLACED' and e.get('wardType')=='CONTROL_WARD']
        paid=None;used=Counter()
        if len(bought)>1: fail(stamp,'multiple_simultaneous_purchases_not_resolved')
        elif bought:
            try: paid,used=recipe_quote(bought[0]['itemId'],inventory,dragon)
            except ValueError: fail(stamp,'unknown_purchase_recipe')
            if destroyed!=used: fail(stamp,'purchase_component_multiset_mismatch',expected=sorted(used.elements()),observed=sorted(destroyed.elements()))
        if len(wards)>1: fail(stamp,'multiple_simultaneous_control_wards')
        for fi,ei,event in entries:
            before=bag();kind=event['type'];item=event.get('itemId') or 0;rule='unsupported'
            if kind=='ITEM_PURCHASED':
                if item==2422: fail(stamp,'explicit_free_boot_grant_not_supported')
                data=dragon.item(item) or {}
                occupancy=physical_slot_state(before,dragon,role=role,bot_quest_complete=False)
                auto=(item in (2138,2139,2140) and data.get('consumed') is True and data.get('consumeOnFull') is True
                      and occupancy['known'] and occupancy['regular_slots']==6)
                if not auto: inventory[item]+=1
                purchases[item].append({'timestamp':stamp,'components':used.copy(),'price':paid,'auto_consumed':auto})
                rule='source_purchase_auto_consumed_elixir' if auto else 'source_purchase'
            elif kind=='ITEM_SOLD':
                if inventory[item]<=0: fail(stamp,'sale_of_unowned_item',item_id=item)
                else: inventory[item]-=1
                sales[item].append({'timestamp':stamp,'refund':dragon.gold_block(item)['sell']})
                rule='source_sale'
            elif kind=='ITEM_UNDO':
                removed,restored=event.get('beforeId') or 0,event.get('afterId') or 0
                if removed:
                    transaction=purchases[removed].pop() if purchases[removed] else None
                    if transaction is None or transaction['auto_consumed'] or inventory[removed]<=0:
                        fail(stamp,'unlinked_purchase_undo')
                    else:
                        inventory[removed]-=1;inventory.update(transaction['components'])
                        if restored and not transaction['components'][restored]: fail(stamp,'undo_restoration_not_original_component')
                        if event.get('goldGain')!=transaction['price']: fail(stamp,'purchase_undo_refund_mismatch')
                    rule='undo_linked_purchase'
                elif restored:
                    sale=sales[restored].pop() if sales[restored] else None
                    if sale is None or event.get('goldGain')!=-sale['refund']: fail(stamp,'unlinked_sale_undo')
                    else: inventory[restored]+=1
                    rule='undo_linked_sale'
                else: fail(stamp,'opaque_undo')
            elif kind=='ITEM_DESTROYED':
                marker=(item==1200 and role=='TOP') or (item==1204 and role=='JUNGLE')
                if marker and item not in quest_seen and not bought:
                    quest_seen.add(item);rule='own_implicit_role_marker'
                else:
                    held=inventory[item]>0
                    if not held: fail(stamp,'destroy_of_unowned_item',item_id=item)
                    else: inventory[item]-=1
                    data=dragon.item(item) or {}
                    if bought and used[item]: rule='source_recipe_consumption'
                    elif held and item in transformations and not bought:
                        inventory[transformations[item]]+=1;rule='pinned_owned_item_transformation'
                    elif item in (2055,772043) and len(wards)==1: rule='paired_control_ward_consumption'
                    elif data.get('consumed') and 'Consumable' in data.get('tags',[]) and item not in (2055,772043):
                        rule='source_consumable_use'
                    else: fail(stamp,'unexplained_nonshop_destruction',item_id=item)
            elif kind=='WARD_PLACED':
                if event.get('wardType')!='CONTROL_WARD': rule='trinket_placement'
                elif destroyed[2055]+destroyed[772043]: rule='ward_destroy_already_accounted'
                else:
                    held=next((i for i in (2055,772043) if inventory[i]>0),None)
                    if held is None: fail(stamp,'ward_placement_without_ownership')
                    else: inventory[held]-=1
                    rule='source_ward_placement_consumption'
            else: fail(stamp,'unsupported_source_event',event_type=kind)
            rules[rule]+=1;ledger.append({'timestamp':stamp,'frame_index':fi,'frame_event_index':ei,
                'rule':rule,'before':before,'after':bag(),'source_event':event})
            boundary_states[stamp].append(bag())
        if stamp in grants:
            grant=grants[stamp];before=bag()
            occupancy=physical_slot_state(before,dragon,role=role,bot_quest_complete=False)
            if (not occupancy['known'] or occupancy['regular_slots']>=6 or occupancy['boot_copies']
                or grant.get('gold_delta')!=0 or grant.get('purchase_cost')!=0 or grant.get('purchase_label') is not False):
                fail(stamp,'grant_conflicts_with_independent_owned_slots')
            inventory[2422]+=1;rules['inferred_footwear_grant_independently_checked']+=1
            ledger.append({'timestamp':stamp,'rule':'inferred_footwear_grant_independently_checked','before':before,'after':bag()})
            boundary_states[stamp].append(bag())
        states_after[stamp]=bag()
    def modeled(values):
        result=Counter(i for i in values if i and not dragon.classify(i)['skip'])
        return result
    times=sorted(states_after);crosschecks=0;skipped=0
    def candidates(stamp,strict=False):
        index=bisect_left(times,stamp)-1
        prior=states_after[times[index]] if index>=0 else []
        if strict or stamp not in boundary_states: return [modeled(prior)]
        return [modeled(v) for v in boundary_states[stamp]]
    for event in replay_events or []:
        if grant_boundary_reason(event,all_grants,guard_ms=1000): skipped+=1;continue
        anchor=event['ts']-15000 if event.get('is_save') else event['ts']
        for member in event['board']:
            if member['participant_id']==owner:
                crosschecks+=1
                if modeled(ids(member['items'])) not in candidates(anchor): fail(anchor,'board_ownership_disagrees_with_source_ledger')
        if event['participant_id']==owner:
            crosschecks+=2
            if modeled(ids(event['inventory_before'])) not in candidates(anchor,strict=True): fail(anchor,'shop_before_disagrees_with_source_ledger')
            if modeled(ids(event['inventory_after'])) not in candidates(event['ts_end']): fail(event['ts_end'],'shop_after_disagrees_with_source_ledger')
    if replay_events is not None and not crosschecks: fail(0,'no_independent_ownership_crosschecks')
    # The final DTO verifies the finished ledger only; it never changes it.
    expected=modeled([actor.get('item'+str(i)) or 0 for i in range(7)])
    actual=modeled(bag())
    for value in (expected,actual):
        for ward in (2055,772043):
            if value[ward]: value[ward]=1
    if actual!=expected: fail(times[-1] if times else 0,'terminal_modeled_inventory_mismatch',extra=sorted((actual-expected).elements()),missing=sorted((expected-actual).elements()))
    return {'version':VERSION,'match_id':match['metadata']['matchId'],'participant_id':owner,'role':role,
        'passed':not failures,'failures':failures,'ledger':ledger,'rule_counts':dict(rules),
        'ownership_crosschecks':crosschecks,'boundary_decisions_not_used_as_inputs':skipped,
        'limits':['No possession inventory is guessed or restored from future data.',
            'Only TOP and JUNGLE role transforms are currently reviewed by this strict gate.',
            'Frame championStats are not model inputs; owner progression and all emitted inventory contexts are checked.',
            'Source consistency does not independently prove the absence of every unobserved possession effect.']}
