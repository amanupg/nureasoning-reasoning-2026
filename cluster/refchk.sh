#!/bin/bash
# usage: refchk.sh <regex>  -- references to the regex in sibling agents' code (read-only)
cd /path/to/cluster/scratch
grep -rnE "$1" --include=*.py --include=*.sh --include=*.sbatch --include=*.yaml --include=*.md \
  --exclude-dir=.venv --exclude-dir=env --exclude-dir=envs --exclude-dir=external --exclude-dir=sif \
  --exclude-dir=.cache --exclude-dir=cache --exclude-dir=logs --exclude-dir=runs --exclude-dir=data --exclude-dir=.uv-python --exclude-dir=sandbox --exclude-dir=site-packages \
  alpasim alpasim-jepa wearable jepa-nureasoning 2>/dev/null | cut -c1-200 | head -20
echo "refchk end"
