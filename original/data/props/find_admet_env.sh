
#!/usr/bin/env bash

set -euo pipefail



module purge

module load anaconda/2023.03

source "$(conda info --base)/etc/profile.d/conda.sh"



echo "Searching conda envs under /work/users/w/a/wanzhou/conda_envs ..."



for py in /work/users/w/a/wanzhou/conda_envs/*/bin/python; do

    [ -x "$py" ] || continue



    "$py" - <<'PY' 2>/tmp/admet_check.err || true

import sys

import importlib.util



mods = ["admet_ai", "rdkit", "torch", "pandas"]

found = {m: importlib.util.find_spec(m) is not None for m in mods}



if found["admet_ai"]:

    print("FOUND_ADMET_ENV")

    print("python:", sys.executable)

    print("version:", sys.version.replace("\n", " "))

    print("mods:", found)

PY



done

