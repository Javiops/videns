"""Validate a real user-submitted rating payload; do not pass UI drafts or QA data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from winprob.m2_quality_review_v1 import strict_json,validate_and_summarize


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(review,cases,output,source_note):
    review=Path(review).resolve(strict=True);cases=Path(cases).resolve(strict=True);output=Path(output).resolve()
    if output.exists():raise FileExistsError('Refuse review receipt overwrite')
    if not source_note.strip():raise ValueError('Document the actual user-message provenance; do not assert a draft is submitted')
    digest=sha(cases)
    if cases.with_suffix('.sha256').read_text().split()!=[digest,cases.name]:raise ValueError('Case sidecar mismatch')
    source_bytes=review.read_bytes();source=source_bytes.decode('utf-8-sig');payload=strict_json(source);public=strict_json(cases.read_text(encoding='utf-8'))
    result=validate_and_summarize(payload,public,digest)
    result.update(source_note=source_note,source_review_sha256=sha(review),source_review_path=str(review),cases_path=str(cases),
        ingestion_code_sha256=sha(__file__),validation_code_sha256=sha(ROOT/'winprob/m2_quality_review_v1.py'))
    output.mkdir(parents=True,exist_ok=False)
    # Preserve exact submitted text separately from its validated interpretation.
    (output/'submitted_review.json').write_bytes(source_bytes)
    path=output/'receipt.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha(path)+' '+path.name+'\n',encoding='ascii')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',required=True,type=Path)
    parser.add_argument('--cases',type=Path,default=ROOT/'artifacts/m2-quality-review/20261001-v1/cases.json')
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--source-note',required=True)
    args=parser.parse_args();result=run(args.review,args.cases,args.output,args.source_note)
    print(json.dumps({k:result[k] for k in ('state','submitted_case_entries','graded_options','cases_with_context_notes')}))
