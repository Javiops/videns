"""Windows child-tree memory enforcement is active before the venv launcher runs."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_patch_development import WindowsMemoryJob


def run(code,limit_mb):
    job=WindowsMemoryJob(limit_mb*1024*1024)
    try:
        child=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,creationflags=0x00000004)
        job.attach(child)
        out,err=child.communicate(timeout=30)
        return child.returncode,out,err,job.peak_bytes()
    finally:
        job.close()


result,out,err,peak=run("print('owned-child-ready')",512)
assert result==0 and out.strip()=='owned-child-ready', (result,out,err)
assert peak>0, 'Real interpreter escaped the memory job'
code="import sys\ntry:\n    data=bytearray(256*1024*1024)\nexcept MemoryError:\n    sys.exit(17)\nsys.exit(0)"
result,out,err,peak=run(code,96)
assert result==17, (result,out,err,peak)
# The reported peak can include a denied allocation; the child receiving
# MemoryError is the enforcement check, not a bound on this diagnostic counter.
assert peak>0
print('ok Windows supervisor: suspended launcher inherits child-tree job, normal child succeeds, aggregate memory cap blocks oversized allocation')
