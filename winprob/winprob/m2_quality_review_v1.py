"""Validate explicitly submitted review data without inventing purchase utility.

This parser cannot authenticate a human author. The caller must establish
that the payload was actually submitted by the user, not copied from a draft
or a synthetic UI test. It never opens M2 ranks, purchase targets or outcomes.
"""
import json
from itertools import combinations

SCHEMA='m2-purchase-quality-user-review-v1'
GRADES={'preferred':2,'acceptable':1,'discouraged':0,'insufficient_information':None}


def need(value,message):
    if not value:raise ValueError(message)


def strict_json(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text,object_pairs_hook=unique,parse_constant=invalid)


def validate_and_summarize(payload,cases,expected_hash):
    need(isinstance(payload,dict) and set(payload)=={'schema','version','cases_sha256','submitted','answers'},'Unexpected review fields')
    need(payload['schema']==SCHEMA and payload['submitted'] is True,'Only an explicit submission may be ingested; drafts are not labels')
    need(cases['schema']=='m2-purchase-quality-blind-cases-v1' and payload['version']==cases['version']
         and payload['cases_sha256']==expected_hash,'Review belongs to another case version')
    catalogue={case['case_id']:[option['label'] for option in case['options']] for case in cases['cases']}
    need(len(catalogue)==len(cases['cases']) and all(len(set(labels))==len(labels) for labels in catalogue.values()),'Duplicate case or option identity')
    answers=payload['answers'];need(isinstance(answers,dict) and set(answers)<=set(catalogue),'Unknown case identity')
    results=[];rated_options=0;notes=0
    for case_id,labels in catalogue.items():
        answer=answers.get(case_id,{'ratings':{},'reason':''})
        need(isinstance(answer,dict) and set(answer)=={'ratings','reason'},'Unexpected case-answer fields')
        ratings=answer['ratings'];reason=answer['reason']
        need(isinstance(ratings,dict) and set(ratings)<=set(labels),'Unknown alternative label')
        need(all(isinstance(v,str) and v in GRADES for v in ratings.values()),'Unknown purchase-quality grade')
        need(isinstance(reason,str) and len(reason)<=500,'Review reason exceeds the declared limit')
        known={label:GRADES[value] for label,value in ratings.items() if GRADES[value] is not None}
        strict=[];same=[]
        for left,right in combinations(labels,2):
            if left not in known or right not in known:continue
            if known[left]==known[right]:same.append([left,right])
            else:
                better,worse=(left,right) if known[left]>known[right] else (right,left)
                strict.append({'preferred_label':better,'less_preferred_label':worse})
        results.append({'case_id':case_id,'ratings':dict(ratings),'reason':reason,'all_options_graded':set(ratings)==set(labels),
            'unrated_labels':[label for label in labels if label not in ratings],
            'insufficient_information_labels':[label for label in labels if ratings.get(label)=='insufficient_information'],
            'strict_preferences_within_declared_rubric':strict,'same_grade_pairs':same})
        rated_options+=len(ratings);notes+=int(bool(reason.strip()))
    need(rated_options>0 or notes>0,'Submission contains no judgement or context note')
    return {'schema':'m2-purchase-quality-review-validation-v1','state':'validated_submission_payload',
        'cases_version':cases['version'],'cases_sha256':expected_hash,'submitted_case_entries':len(answers),
        'graded_options':rated_options,'cases_with_context_notes':notes,'cases':results,
        'source_origin_authenticated_by_parser':False,'original_m2_ranks_read':False,'historical_targets_read':False,
        'outcomes_read':False,'models_trained':False,'performance_improvement_measured':False,
        'limits':['Caller must independently establish an actual user submission; schema fields alone do not authenticate its author.',
                  'The selected pilot establishes a rubric, not representative model accuracy or causal item value.',
                  'Equal grades are coarse rubric ties, not a proof of equal utility; unknowns create no preference edge.',
                  'Notes may state assumptions or missing intent and remain data, never executable instructions.',
                  'No original M2 ranking is exposed by this validation step.']}
