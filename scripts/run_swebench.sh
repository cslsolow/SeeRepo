#!/usr/bin/env bash
# Run SeeRepo on SWE-bench Verified.
#
# Prerequisites:
#   1. Build (or download) the graph index:
#        python scripts/build_graph_index.py \
#            --dataset princeton-nlp/SWE-Bench_Verified \
#            --split test \
#            --output-dir /path/to/graph_index \
#            --workers 8
#   2. Set the environment variables below.
#   3. Ensure Docker is running and the SWE-bench images are accessible.
#
# Usage:
#   export SEEREPO_GRAPH_INDEX_DIR=/path/to/graph_index
#   export OPENAI_API_KEY=<key>   # if using OpenAI / compatible endpoints
#   # and/or ANTHROPIC_API_KEY, etc., depending on model in your YAML
#   bash scripts/run_swebench.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${SEEREPO_GRAPH_INDEX_DIR:?Please set SEEREPO_GRAPH_INDEX_DIR to the graph index directory}"

if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" && -z "${AZURE_API_KEY:-}" ]]; then
  echo "Warning: No OPENAI_API_KEY, ANTHROPIC_API_KEY, or AZURE_API_KEY set. LiteLLM may fail at runtime." >&2
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/trajectories/seerepo_verified}"
WORKERS="${WORKERS:-4}"
CONFIG="${CONFIG:-${REPO_ROOT}/src/minisweagent/config/extra/SeeRepo.yaml}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export SEEREPO_GRAPH_INDEX_DIR

mkdir -p "${OUTPUT_DIR}"

echo "Graph index : ${SEEREPO_GRAPH_INDEX_DIR}"
echo "Config      : ${CONFIG}"
echo "Output      : ${OUTPUT_DIR}"
echo "Workers     : ${WORKERS}"
echo ""

python -m minisweagent.run.extra.swebench \
    --subset verified \
    --split test \
    --config "${CONFIG}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR}"
