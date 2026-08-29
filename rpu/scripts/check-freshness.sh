#!/usr/bin/env bash
# PARANOIA rule 5: a claim about a configuration is only valid against a simulator built
# after the last change to the kernel or the plans.
#
# Compares against source file MTIME, not git commit time -- a commit that only touches
# docs still moves the commit timestamp, which would flag a perfectly current binary.
set -uo pipefail
CY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CY" || exit 1
SRC=(rpu/experiments/rpu_gemm.py generators/chipyard/src/main/scala/rpu/GemmExecPlan.scala
     generators/chipyard/src/main/scala/config/FSAConfig.scala)
newest=0
for f in "${SRC[@]}"; do
  m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  [ "$m" -gt "$newest" ] && newest=$m
done
echo "newest kernel/plan source: $(date -d @$newest '+%F %T')"
stale=0
for sim in sims/verilator/simulator-chipyard.harness-*; do
  [ -x "$sim" ] || continue
  name=${sim##*harness-}
  b=$(stat -c %Y "$sim")
  if [ "$b" -gt "$newest" ]; then
    printf "  fresh  %-30s %s\n" "$name" "$(date -d @$b '+%F %T')"
  else
    printf "  STALE  %-30s %s\n" "$name" "$(date -d @$b '+%F %T')"; stale=1
  fi
done
[ $stale -eq 0 ] && echo "all simulators current" || echo "REBUILD the STALE ones before claiming anything about them"
exit $stale
