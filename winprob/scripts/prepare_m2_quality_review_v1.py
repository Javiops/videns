"""Prepare an outcome-blind rubric pilot from existing admitted M2 artifacts."""
from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.iql_dataset_contract import bound_json,current_reservations,file_hash

BASE=ROOT.parent/'build_tracker/data/exports/winprob-ordinary'
OUT=ROOT/'artifacts/m2-quality-review/20261001-v1'
STATIC=ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'


def need(value,message):
    if not value:raise ValueError(message)


def value_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
    Path(path).with_suffix('.sha256').write_text(file_hash(path)+' '+Path(path).name+'\n',encoding='ascii')


class PlainText(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)
    def handle_starttag(self,tag,attrs):
        if tag in ('br','p','li','hr'):self.parts.append(' ')


def run():
    need(not OUT.exists(),'Refuse review version overwrite');bindings={}
    cm,ch=bound_json(BASE/'ordinary-contexts-20260930-v1/manifest.json',bindings)
    rm,rh=bound_json(BASE/'ordinary-reference-20260930-v3/manifest.json',bindings)
    am,ah=bound_json(BASE/'ordinary-native19-20260930-v3/manifest.json',bindings)
    admission,adsha=bound_json(cm['admission_path'],bindings);need(adsha==cm['admission_sha256'],'Admission changed')
    allowed={r['match_id'] for r in admission['games'] if r['role']=='development' and r['patch']=='16.19'}
    need(len(allowed)==859 and set(cm['admitted_ids'])==allowed and not allowed&current_reservations(admission),'Unsafe source scope')
    need(all(m['state']=='complete' and not m['test_payload_reads'] and not m['confirmation_payload_reads'] for m in (cm,rm,am)),'Incomplete or unsafe sources')
    need(rm['source_manifest_sha256']==ch and am['source_manifest_sha256']==ch and am['reference_manifest_sha256']==rh,'Source chain mismatch')
    need(file_hash(STATIC)=='5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0','Changed public item export')
    items=json.loads(STATIC.read_text())['patches']['16.19']['items'];bindings[str(STATIC)]=file_hash(STATIC)
    paths=[BASE/'ordinary-contexts-20260930-v1/contexts.jsonl',BASE/'ordinary-reference-20260930-v3/rows.jsonl',BASE/'ordinary-native19-20260930-v3/rows.jsonl']
    for path,digest in zip(paths,(cm['files']['contexts.jsonl'],rm['files']['rows.jsonl'],am['files']['rows.jsonl']),strict=True):
        need(file_hash(path)==digest,'Changed frozen context/options');bindings[str(path)]=digest
    bindings[str(Path(__file__))]=file_hash(__file__);OUT.mkdir(parents=True)
    plan={'schema':'m2-purchase-quality-rubric-pilot-plan-v1','purpose':'Define a purchase-quality rubric; not a performance estimate or experiment winner.',
        'selection_published_before_option_content_review':True,'schema_keys_of_one_admitted_row_previously_inspected':True,
        'source_scope_games':859,'requested_cases':8,'eligibility':'Available16.19context;8<=minute<=25;at least2M2buy options; every shownM2option native19feasible; known item names; no unmodeled focal slots.',
        'ranking':'SHA256(m2-quality-review-v1/ + sample_id), at most one case per focal champion; stop after8; no target/outcome selection.',
        'blind_option_order':'SHA256(m2-quality-review-v1/options/ + sample_id + / + group). Original M2 ranks stored in separate escrow only.',
        'rating_categories':['preferred','acceptable','discouraged','insufficient_information'],'ties_allowed':True,
        'requires_independent_judgement':True,'target_file_read':False,'outcome_read':False,'new_raw_read':False,'models_scored_or_trained':False,
        'bindings_sha256':bindings,'limits':['Selected exploratory rubric cases, not a representative performance sample.',
            'Budget is the supplied predecision estimate, not an observed live exact wallet.',
            'M2 used its frozen16.18runtime; every displayed option has a separate native16.19cost/legality receipt.',
            'No historical purchase, game result, M2probability, M2rank, personal identifier or inferred role is shown.',
            'Missing tactical intent and timing must be marked as insufficient information, never filled with an invented best label.']}
    write(OUT/'plan.json',plan)
    def item_rows(values):
        counts=Counter(int(x) for x in values if int(x)!=0)
        need(all(str(i) in items for i in counts),'Unknown public item ID')
        return [{'item_id':i,'name':items[str(i)]['name'],'count':n} for i,n in sorted(counts.items())]
    eligible=[];exclusions=Counter();seen=set()
    with paths[0].open(encoding='utf-8') as cf,paths[1].open(encoding='utf-8') as rf,paths[2].open(encoding='utf-8') as af:
        for ordinal,(cl,rl,al) in enumerate(zip(cf,rf,af,strict=True)):
            c,r,a=(json.loads(v) for v in (cl,rl,al));mid=c['match_id'];sid=c['sample_id']
            need(mid in allowed and sid not in seen,'Unexpected or duplicate source row');seen.add(sid)
            for row in (r,a):
                need(row['sample_id']==sid and row['match_id']==mid and row['ordinal']==ordinal and row['projection_sha256']==c['projection_sha256'],
                     'Context/option/native-audit alignment differs')
            need(c['ordinal']==ordinal and c['status']=='available' and c['role']=='development' and c['patch']=='16.19','Unexpected source status')
            s=c['prestate'];minute=s['ts']/60000;options=r['m2_options']
            if not 8<=minute<=25:exclusions['outside_predeclared_window']+=1;continue
            if sum(o['kind']=='buy' for o in options)<2:exclusions['fewer_than2buy_options']+=1;continue
            if s['unmodeled_regular_slots']!=0:exclusions['unmodeled_focal_slots']+=1;continue
            legal={x['option_sha256']:x for x in a['candidates'] if 'm2' in x['arms']}
            need(len(legal)==len(options) and all(legal[value_hash(o)]['feasible'] for o in options),'Incomplete native M2 legality')
            try:
                focal={'champion':s['champion'],'level':s['level'],'inventory':item_rows(s['inventory']),
                    'kills':s['kills'],'deaths':s['deaths'],'assists':s['assists'],'cs':s['cs']}
                allies=[];enemies=[]
                for player in s['others']:
                    display={'champion':player['champion'],'level':player['level'],'inventory':item_rows(player['items'])}
                    (allies if player['team_id']==s['team_id'] else enemies).append(display)
                need(len(allies)==4 and len(enemies)==5,'Incomplete team view')
                presented=[];escrow=[]
                ordered=sorted(enumerate(options),key=lambda pair:hashlib.sha256(('m2-quality-review-v1/options/'+sid+'/'+str(pair[1]['group'])).encode()).digest())
                for j,(original_rank,opt) in enumerate(ordered):
                    key=chr(65+j);proof=legal[value_hash(opt)]
                    need(proof['option']==opt and proof['arms']['m2']==original_rank,'Native action identity/rank differs')
                    cost=float(proof['incremental_cost']);need(0<=cost<=s['gold_est']+1e-6,'Invalid supplied-wallet cost')
                    bought=item_rows([int(i) for i,count in opt['items'].items() for _ in range(count)])
                    details=[]
                    for item in bought:
                        parser=PlainText();parser.feed(items[str(item['item_id'])].get('description',''))
                        details.append({'name':item['name'],'description':' '.join(' '.join(parser.parts).split())})
                    presented.append({'label':key,'kind':opt['kind'],'items':bought,'incremental_cost':cost,'remaining_gold':s['gold_est']-cost,'item_details':details})
                    escrow.append({'label':key,'group':opt['group'],'original_m2_rank':original_rank,'option_sha256':value_hash(opt),'native19_sequence':proof['sequence']})
            except ValueError as error:
                if str(error)=='Unknown public item ID':exclusions['unknown_item_name']+=1;continue
                raise
            public={'patch':'16.19','timestamp_ms':s['ts'],'budget':s['gold_est'],'budget_kind':'estimated_context_wallet',
                'focal':focal,'allies':allies,'enemies':enemies,'ally_objectives':s['ally_obj'],'enemy_objectives':s['enemy_obj'],
                'options':presented,'footwear_granted':s['footwear_granted'],'magical_footwear_selected':s['magical_footwear_selected'],
                'bot_quest_complete':s['bot_quest_complete']}
            private={'sample_id':sid,'match_id':mid,'ordinal':ordinal,'source_projection_sha256':c['projection_sha256'],'options':escrow}
            rank=hashlib.sha256(('m2-quality-review-v1/'+sid).encode()).hexdigest();eligible.append((rank,public,private))
    need(len(seen)==859,'Source universe incomplete');picked=[];used=set()
    for rank,public,private in sorted(eligible,key=lambda row:row[0]):
        if public['focal']['champion'] in used:continue
        used.add(public['focal']['champion']);picked.append((rank,public,private))
        if len(picked)==8:break
    need(len(picked)==8,'Insufficient cases; do not relax selection')
    cases=[];escrow=[]
    for i,(rank,public,private) in enumerate(picked,1):
        case_id='C'+str(i).zfill(2);cases.append({'case_id':case_id,**public});escrow.append({'case_id':case_id,'selection_sha256':rank,**private})
    need(not {r['match_id'] for r in escrow}&current_reservations(admission),'Reservation changed')
    for path,digest in bindings.items():need(file_hash(path)==digest,'Bound input changed while preparing review')
    write(OUT/'cases.json',{'schema':'m2-purchase-quality-blind-cases-v1','version':OUT.name,'cases':cases,'limits':plan['limits']})
    write(OUT/'escrow.json',{'schema':'m2-purchase-quality-review-escrow-v1','cases':escrow,'not_for_blind_review':True})
    write(OUT/'manifest.json',{'schema':'m2-purchase-quality-rubric-pilot-v1','state':'prepared_unrated','source_rows':859,
        'eligible_rows':len(eligible),'selected_cases':8,'exclusions':dict(exclusions),'files':{n:file_hash(OUT/n) for n in ('plan.json','cases.json','escrow.json')},
        'target_file_read':False,'outcomes_read':False,'raw_payload_reads':False,'models_scored_or_trained':False,'reserved_intersections':0,
        'quality_labels_created':0,'performance_improvement_measured':False,'bindings_sha256':bindings})
    print(json.dumps({'state':'prepared_unrated','eligible_rows':len(eligible),'selected_cases':8,'exclusions':dict(exclusions),'output':str(OUT)},indent=2))


if __name__=='__main__':run()
