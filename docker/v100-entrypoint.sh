#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -eq 0 || $1 == -* ]]; then
  if [[ "${SGLANG_V100_SKIP_STARTUP_CHECK:-0}" != 1 ]]; then
    bash /opt/sglang/scripts/smoke_v100.sh
  fi
  exec sglang serve "$@"
fi

exec "$@"
