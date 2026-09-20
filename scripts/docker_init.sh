#!/usr/bin/env bash
# =============================================================================
# scripts/docker_init.sh — One-shot ICSR stack initialisation (Linux / macOS)
# =============================================================================
# Run ONCE after the first `docker compose up -d` to:
#   1. Pull the Ollama model into the container
#   2. Wait for the HITL server to become healthy
#   3. Create an initial reviewer API key
#
# Usage:
#   chmod +x scripts/docker_init.sh
#   ./scripts/docker_init.sh
#
# Optional env vars:
#   OLLAMA_MODEL      — Ollama model to pull (default: llama3.1:8b-instruct-q4_K_M)
#   REVIEWER_ID       — Reviewer ID for initial key (default: QPPV-01)
#   REVIEWER_ROLE     — Role (default: QPPV)
#   HITL_PORT         — HITL server port (default: 8000)
# =============================================================================
set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3.1:8b-instruct-q4_K_M}"
REVIEWER_ID="${REVIEWER_ID:-QPPV-01}"
REVIEWER_ROLE="${REVIEWER_ROLE:-QPPV}"
HITL_PORT="${HITL_PORT:-8000}"
MAX_WAIT=120

echo "╔══════════════════════════════════════════════════════╗"
echo "║   ICSR Stack Init                                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Pull Ollama model ────────────────────────────────────────────────
echo "[1/3] Pulling Ollama model: $MODEL …"
docker compose exec -T ollama ollama pull "$MODEL"
echo "      ✓ Model ready"
echo ""

# ── Step 2: Wait for HITL server health ──────────────────────────────────────
echo "[2/3] Waiting for HITL server (max ${MAX_WAIT}s) …"
elapsed=0
until curl -sf "http://localhost:${HITL_PORT}/health" > /dev/null 2>&1; do
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "      ✗ Timed out waiting for HITL server"
        echo "        Check logs: docker compose logs icsr-hitl"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    printf "."
done
echo ""
echo "      ✓ HITL server healthy"
echo ""

# ── Step 3: Create initial reviewer ──────────────────────────────────────────
echo "[3/3] Creating initial reviewer: $REVIEWER_ID (role=$REVIEWER_ROLE) …"
API_KEY=$(docker compose exec -T icsr-hitl \
    python scripts/manage_reviewers.py add \
        --reviewer-id "$REVIEWER_ID" \
        --role "$REVIEWER_ROLE" 2>&1 | grep "api_key:" | awk '{print $2}')

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Initialisation complete!                           ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║   Reviewer ID : $REVIEWER_ID"
echo "║   API Key     : $API_KEY"
echo "║                                                      ║"
echo "║   ⚠  Save this key — it is shown exactly once.      ║"
echo "║                                                      ║"
echo "║   Test:                                              ║"
printf "║   curl -H 'X-API-Key: %s' \\\\\n" "$API_KEY"
printf "║        http://localhost:%s/health\n" "$HITL_PORT"
echo "╚══════════════════════════════════════════════════════╝"
