"""Launch the read-only diagnostic in both original CPU environments."""
import subprocess
from common import *

require_allocation()
verify()
directory = ROOT / 'property_exception_diagnosis_20261011' / os.environ['SLURM_JOB_ID']
directory.mkdir(parents=True, exist_ok=False)
for mode, envname in [('inf', 'gellmo_inf'), ('standard', 'gellmo_eval_py36')]:
    env = isolate(directory / mode, cpu=True)
    command = [ENVS[envname], '-B', str(ROOT / 'diagnose_s1_property_exception.py'),
               '--mode', mode, '--output', str(directory / (mode + '.json'))]
    subprocess.run(command, env=env, check=True)
