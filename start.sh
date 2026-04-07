#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec uv run kennel \
  --port 9000 \
  --secret-file ~/.kennel-secret \
  --self-repo rhencke/kennel \
  rhencke/confusio:/home/rhencke/workspace/confusio \
  rhencke/kennel:/home/rhencke/workspace/kennel
