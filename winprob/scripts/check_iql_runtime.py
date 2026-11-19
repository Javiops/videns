"""Tiny synthetic CUDA/autograd check in this project's own environment."""
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError('Own runtime has no available CUDA device')
torch.manual_seed(20260930)
weight = torch.randn(128, 32, device='cuda', requires_grad=True)
x = torch.randn(64, 128, device='cuda')
loss = (x @ weight).square().mean()
loss.backward()
torch.cuda.synchronize()
if not torch.isfinite(loss) or not torch.isfinite(weight.grad).all():
    raise RuntimeError('Synthetic CUDA gradient is nonfinite')
report = {'state': 'passed', 'python': sys.executable, 'torch': torch.__version__,
          'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'synthetic_loss': loss.item(), 'game_payload_reads': False, 'model_training': False}
out = Path(__file__).resolve().parents[1]/'data/reports/iql_runtime_20260930.json'
if out.exists():
    raise RuntimeError('Runtime probe record already exists')
out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
