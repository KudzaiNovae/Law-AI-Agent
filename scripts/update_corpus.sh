#!/usr/bin/env bash
# Refresh the corpus: scrape new gazettes, rebuild the index, redeploy.
#
# Run weekly on your machine (crontab -e):
#   0 6 * * 1  cd /home/kudzai/Documents/law-agent && ./scripts/update_corpus.sh >> /tmp/corpus_update.log 2>&1
#
# Requires: .venv, docker, az CLI logged in.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
ACR=onegovacr
IMAGE=$ACR.azurecr.io/onegov:latest
APP=onegov            # container app name
RG=onegov-rg          # resource group — adjust if yours differs

echo "== $(date -Is) ingest =="
$PY -m ingest.run

echo "== rebuild index =="
$PY -m index.build_index

echo "== rebuild knowledge graph =="
# LLM extraction is cached per file; only new instruments hit the API.
$PY -m graph.build_graph || $PY -m graph.build_graph --no-llm

echo "== rebuild + push image =="
docker build -t "$IMAGE" .
az acr login -n "$ACR"
docker push "$IMAGE"

echo "== redeploy =="
# unique revision suffix forces ACA to pull the new :latest image
az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" \
  --revision-suffix "r$(date +%Y%m%d%H%M)"

echo "== done $(date -Is) =="
