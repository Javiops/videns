"""Synthetic-only full-catalogue student training-memory/time measurement."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from winprob.causal_student_v1 import CausalStudent, purchase_objective, KEYSTONES, STYLES
from winprob.iql_models import PublicBoardBatch
from winprob.iql_training import memory_snapshot
from winprob.iql_dataset_contract import file_hash


def run(output):
    started = time.monotonic(); output = output.resolve()
    if output.exists(): raise ValueError('immutable new synthetic run required')
    output.mkdir(parents=True)
    paths = [Path(__file__), ROOT/'docs/causal_student_resource_protocol_v1.md', ROOT/'docs/causal_student_design_v1.md',
             ROOT/'winprob/causal_student_v1.py', ROOT/'winprob/iql_models.py', ROOT/'winprob/iql_training.py']
    code = {str(p.resolve()): file_hash(p) for p in paths}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261004)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available(): raise RuntimeError('idle CUDA required')
    device = torch.device('cuda'); before = memory_snapshot()
    free_before, total_before = torch.cuda.mem_get_info(); reserved_before = torch.cuda.memory_reserved()
    teacher_reservation = torch.empty((56288, 9333), dtype=torch.float32, device=device)
    model = CausalStudent(180, 1024).to(device).train()
    b = 128
    board = PublicBoardBatch(torch.randint(1, 181, (b, 10), device=device),
        torch.randint(1, 1025, (b, 10, 12), device=device), torch.ones(b, 10, 12, device=device),
        torch.randn(b, 10, 22, device=device), torch.ones(b, 10, 1, device=device),
        torch.tensor([[False]+[True]*4+[False]*5]*b, device=device),
        torch.tensor([[False]*5+[True]*5]*b, device=device), torch.randn(b, 37, device=device))
    runes = torch.stack((torch.randint(len(KEYSTONES)+1, (b,), device=device), torch.randint(len(STYLES)+1, (b,), device=device)), 1)
    raw = torch.arange(9333, device=device)
    sizes = raw % 6+1; sizes[0] = 0
    positions = torch.arange(6, device=device)[None]
    ids = (raw[:, None]*7+positions) % 1024+1; counts = (positions < sizes[:, None]).to(torch.float32); ids[counts == 0] = 0
    attributes = torch.rand(9333, 23, device=device); attributes[0] = 0; attributes[:, 22] = (raw == 0).float()
    empty = raw == 0; target = torch.zeros(b, dtype=torch.long, device=device)
    teacher = torch.full((b, 9333), 1/9333, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats(); timings = []; losses = []
    for step in range(6):
        if time.monotonic()-started >= 900: raise RuntimeError('15minute synthetic resource bound')
        torch.cuda.synchronize(); t = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        logits = model(board, runes, ids.long(), counts, attributes, empty, chunk=512, checkpoint_deep=True)
        loss, pieces = purchase_objective(logits, target, teacher); loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite synthetic student gradient')
        optimizer.step(); torch.cuda.synchronize(); seconds = time.monotonic()-t
        losses.append({'step': step, 'synthetic_total_loss': float(loss.detach()),
                       'synthetic_hard_loss': float(pieces['hard'].detach()), 'synthetic_teacher_KL': float(pieces['teacher_KL'].detach())})
        if step >= 2: timings.append(seconds)
        print(json.dumps({'stage': 'synthetic_resource_probe', 'completed': step+1, 'seconds': seconds}), flush=True)
    free, total = torch.cuda.mem_get_info(); peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    projected_free_at_peak = free_before-(peak_reserved-reserved_before)
    for path, sha in code.items():
        if file_hash(path) != sha: raise ValueError('prototype changed during resource pilot')
    result = {'schema': 'causal-student-synthetic-resource-v1', 'state': 'complete',
        'synthetic_batch': 128, 'synthetic_actions': 9333, 'action_chunk': 512,
        'teacher_reservation_shape': [56288, 9333], 'teacher_reservation_bytes': teacher_reservation.numel()*teacher_reservation.element_size(),
        'parameters': sum(p.numel() for p in model.parameters()), 'warmup_steps': 2, 'timed_steps': 4,
        'timings_s': timings, 'mean_step_s': sum(timings)/len(timings),
        'peak_cuda_allocated_bytes_including_reservation': peak,
        'peak_cuda_reserved_bytes': peak_reserved, 'free_cuda_bytes_at_end': free,
        'free_cuda_bytes_before': free_before, 'projected_free_after_peak_reservation': projected_free_at_peak,
        'device_total_bytes': total, 'one_GiB_device_headroom_at_peak': bool(min(free, projected_free_at_peak) >= 1024**3),
        'host_before': before, 'host_after': memory_snapshot(), 'synthetic_losses': losses,
        'game_payload_reads': False, 'actual_teacher_probabilities_read': False, 'real_training_performed': False,
        'synthetic_optimizer_steps': 6, 'test_payload_reads': False, 'confirmation_payload_reads': False,
        'accuracy_computed': False, 'checkpoint_saved': False, 'code_sha256': code,
        'torch_version': torch.__version__, 'device_name': torch.cuda.get_device_name(0), 'elapsed_s': time.monotonic()-started,
        'limits': ['Synthetic feasibility only; no actual-data runtime, comparison, recommendation latency or accuracy gain.']}
    path = output/'report.json'; path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    path.with_suffix('.sha256').write_text(file_hash(path)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'mean_step_s', 'peak_cuda_allocated_bytes_including_reservation', 'one_GiB_device_headroom_at_peak', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
