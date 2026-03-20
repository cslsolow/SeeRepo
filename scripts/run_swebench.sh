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
#   export ANTHROPIC_API_KEY=<your-key>
#   export SEEREPO_GRAPH_INDEX_DIR=/path/to/graph_index
#   bash scripts/run_swebench.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${SEEREPO_GRAPH_INDEX_DIR:?Please set SEEREPO_GRAPH_INDEX_DIR to the graph index directory}"
: "${ANTHROPIC_API_KEY:?Please set ANTHROPIC_API_KEY}"

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
