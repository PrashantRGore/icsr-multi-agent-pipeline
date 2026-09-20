#Requires -Version 5.1
# =============================================================================
# scripts\docker_init.ps1 — One-shot ICSR stack initialisation (Windows)
# =============================================================================
# Run ONCE after the first `docker compose up -d` to:
#   1. Pull the Ollama model into the container
#   2. Wait for the HITL server to become healthy
#   3. Create an initial reviewer API key
#
# Usage (PowerShell):
#   .\scripts\docker_init.ps1
#
# Optional env vars:
#   $env:OLLAMA_MODEL   — Ollama model to pull (default: llama3.1:8b-instruct-q4_K_M)
#   $env:REVIEWER_ID    — Reviewer ID for initial key (default: QPPV-01)
#   $env:REVIEWER_ROLE  — Role (default: QPPV)
#   $env:HITL_PORT      — HITL server port (default: 8000)
# =============================================================================
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Model      = if ($env:OLLAMA_MODEL)   { $env:OLLAMA_MODEL }   else { "llama3.1:8b-instruct-q4_K_M" }
$ReviewerId = if ($env:REVIEWER_ID)    { $env:REVIEWER_ID }    else { "QPPV-01" }
$Role       = if ($env:REVIEWER_ROLE)  { $env:REVIEWER_ROLE }  else { "QPPV" }
$Port       = if ($env:HITL_PORT)      { $env:HITL_PORT }      else { "8000" }
$MaxWait    = 120

Write-Host "╔══════════════════════════════════════════════════════╗"
Write-Host "║   ICSR Stack Init                                    ║"
Write-Host "╚══════════════════════════════════════════════════════╝"
Write-Host ""

# ── Step 1: Pull Ollama model ────────────────────────────────────────────────
Write-Host "[1/3] Pulling Ollama model: $Model …"
docker compose exec -T ollama ollama pull $Model
Write-Host "      ✓ Model ready"
Write-Host ""

# ── Step 2: Wait for HITL server health ──────────────────────────────────────
Write-Host "[2/3] Waiting for HITL server (max ${MaxWait}s) …"
$Elapsed = 0
$Healthy = $false
while ($Elapsed -lt $MaxWait) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            $Healthy = $true
            break
        }
    } catch {
        # Not ready yet
    }
    Start-Sleep -Seconds 2
    $Elapsed += 2
    Write-Host -NoNewline "."
}
Write-Host ""

if (-not $Healthy) {
    Write-Error "Timed out waiting for HITL server. Check: docker compose logs icsr-hitl"
    exit 1
}
Write-Host "      ✓ HITL server healthy"
Write-Host ""

# ── Step 3: Create initial reviewer ──────────────────────────────────────────
Write-Host "[3/3] Creating initial reviewer: $ReviewerId (role=$Role) …"
$Output = docker compose exec -T icsr-hitl `
    python scripts/manage_reviewers.py add `
    --reviewer-id $ReviewerId `
    --role $Role 2>&1

$ApiKey = ($Output | Select-String "api_key:").ToString().Split(":")[1].Trim()

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗"
Write-Host "║   Initialisation complete!                           ║"
Write-Host "╠══════════════════════════════════════════════════════╣"
Write-Host "║                                                      ║"
Write-Host "║   Reviewer ID : $ReviewerId"
Write-Host "║   API Key     : $ApiKey"
Write-Host "║                                                      ║"
Write-Host "║   ⚠  Save this key — it is shown exactly once.      ║"
Write-Host "║                                                      ║"
Write-Host "║   Test:                                              ║"
Write-Host "║   Invoke-WebRequest -Uri http://localhost:$Port/health ``"
Write-Host "║     -Headers @{'X-API-Key'='$ApiKey'}"
Write-Host "╚══════════════════════════════════════════════════════╝"
