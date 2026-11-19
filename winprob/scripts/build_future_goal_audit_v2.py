"""Derive a new auditor preserving v1 and adding full original-batch GPU replay."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/reports/future_goal_result_audit_v1.py'
target=ROOT/'data/reports/future_goal_result_audit_v2.py'
if hashlib.sha256(source.read_bytes()).hexdigest()!='6254e31af887e5b6665e592e244cc0c75dcf2fa1f80aa7ace705a76795abe7e6':
    raise ValueError('Original auditor changed')
if target.exists():raise ValueError('Refuse v2 auditor overwrite')
text=source.read_text(encoding='utf-8')

def replace(old,new):
    global text
    if text.count(old)!=1:raise ValueError('Expected one frozen source fragment: '+old[:70])
    text=text.replace(old,new,1)

replace('import argparse\n','import argparse\nimport dataclasses\n')
replace("    bind(__file__,sha(__file__));bind(LEGACY_PATH,sha(LEGACY_PATH));guard()",
"""    bind(__file__,sha(__file__));bind(LEGACY_PATH,sha(LEGACY_PATH));guard()
    original=ROOT/'data/reports/future_goal_result_audit_v1.py'
    bind(original,'6254e31af887e5b6665e592e244cc0c75dcf2fa1f80aa7ace705a76795abe7e6')
    diagnostic=bound(ROOT/'data/reports/future_goal_numerical_diagnostic_v1.json')
    check(diagnostic['original_cpu_threshold_failed'] is True and
          diagnostic['gpu_original_batch_all_logp_bitwise_equal'] is True and
          diagnostic['gpu_original_batch_all_top3_equal'] is True,'Missing diagnosed v1 discrepancy')""")
replace("    checkpoint_checks={};spots=[];near_ties=[];max_error=0.",
"""    checkpoint_checks={};spots=[];near_ties=[];max_error=0.
    cpu_tolerance_exceedances=[];gpu_replays={}
    global_inventory_width=int(np.max(np.diff(a['inventory_offsets'])))
    check(torch.cuda.is_available(),'Full original-batch replay requires idle CUDA')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False""")
replace("            check(np.all(error<=2e-5),'CPU target logP differs beyond2e-5')",
"""            # Preserve and report v1's failed CPU threshold; do not call it passed.
            cpu_tolerance_exceedances.extend({'fold':fold,'arm':arm,'game_id':str(prediction['game_ids'][chosen[i]]),
                'absolute_target_logp_error':float(error[i])} for i in np.flatnonzero(error>2e-5))""")
replace("                    check(max(gaps)<=2e-5,'CPU rank discrepancy is not a numeric near-tie')",
"                    # CPU rank differences are recorded; exact GPU replay below is mandatory.")
replace("            del model,saved,logits;gc.collect()",
"""            # Independent manual public-state gathering, full original batch geometry.
            model=model.to('cuda:0').eval();replayed=0;max_gpu_error=0.
            for offset in range(0,len(positions),256):
                guard();block=positions[offset:offset+256];bstate,bpatch=gather(selected[block])
                padding=global_inventory_width-bstate.inventory_indices.shape[-1]
                check(padding>=0,'Manual bags exceed fixed original inventory width')
                bstate=dataclasses.replace(bstate,
                    inventory_indices=torch.nn.functional.pad(bstate.inventory_indices,(0,padding)),
                    inventory_counts=torch.nn.functional.pad(bstate.inventory_counts,(0,padding)))
                bstate=dataclasses.replace(bstate,**{field:getattr(bstate,field).to('cuda:0') for field in bstate.__dataclass_fields__})
                with torch.inference_mode():full=model(bstate,bpatch.to('cuda:0'))['main_logits'].cpu().numpy().astype(np.float64)
                shifted_gpu=full-full.max(1,keepdims=True)
                replay_logp=shifted_gpu[np.arange(len(block)),prediction['labels'][block]]-np.log(np.exp(shifted_gpu).sum(1))
                replay_top=np.argsort(-full,axis=1,kind='stable')[:,:3]
                expected_logp=prediction[arm+'_target_log_probability'][block]
                max_gpu_error=max(max_gpu_error,float(np.max(np.abs(replay_logp-expected_logp))))
                check(np.array_equal(replay_logp,expected_logp),'Original-batch GPU target logP is not bitwise identical')
                check(np.array_equal(replay_top,prediction[arm+'_top3'][block]),'Original-batch GPU top3 differs')
                replayed+=len(block)
            gpu_replays[name]={'selected_rows':replayed,'all_target_logp_bitwise_equal':True,'all_top3_equal':True,
                               'maximum_abs_target_logp_error':max_gpu_error,'global_inventory_width':global_inventory_width}
            del bstate,bpatch,full,shifted_gpu,replay_logp,replay_top
            del model,saved,logits;gc.collect();torch.cuda.empty_cache()""")
replace("    check(len(checkpoint_checks)==6 and sum(row['steps'] for row in checkpoint_checks.values())==92040,'Incomplete six-model recipe')",
"""    check(len(checkpoint_checks)==6 and sum(row['steps'] for row in checkpoint_checks.values())==92040,'Incomplete six-model recipe')
    check(len(gpu_replays)==6 and sum(row['selected_rows'] for row in gpu_replays.values())==10050,'Full GPU replay incomplete')""")
replace("'schema':'future-goal-independent-result-audit-v1'","'schema':'future-goal-independent-result-audit-v2'")
replace("        'CPU_reinference_rows':96,'CPU_spotchecks':spots,'maximum_abs_CPU_target_logp_error':max_error,'CPU_rank_near_ties':near_ties,",
"""        'CPU_reinference_rows':96,'CPU_spotchecks':spots,'maximum_abs_CPU_target_logp_error':max_error,'CPU_rank_differences':near_ties,
        'original_CPU_2e5_tolerance_passed':not cpu_tolerance_exceedances and not near_ties,
        'CPU_2e5_tolerance_exceedances':cpu_tolerance_exceedances,'GPU_original_batch_replays':gpu_replays,
        'GPU_independent_reinference_rows':10050,'GPU_all_target_logp_bitwise_equal':True,'GPU_all_top3_equal':True,""")
replace("'training_performed':False,'gpu_used':False,","'training_performed':False,'gpu_used':True,")
replace("'Every prediction and metric checked; independent CPU neural inference covers16 fixed rows per fold and arm.',",
"""'Every prediction and metric checked; all10050 neural predictions replayed with independent manual inputs and original GPU batch geometry.',
            'CPU spotcheck differences beyond the original2e-5 threshold are explicitly reported; v1 remains failed and unchanged.',""")
replace("default=ROOT/'data/reports/future_goal_probe_v1_audit.json'","default=ROOT/'data/reports/future_goal_probe_v2_audit.json'")
replace('Final-only independent future-supervision audit; never inspect partial fits.',
        'Final-only v2 audit with exact full GPU replay; v1 CPU threshold failure is retained.')
target.write_text(text,encoding='utf-8')
print(str(target))
