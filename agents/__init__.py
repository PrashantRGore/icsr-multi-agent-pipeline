"""
agents/__init__.py
==================
Phase 3 — Agent package.

All agents follow the same LangGraph node contract:
    def run(state: GraphState) -> dict
Returns a partial GraphState dict; LangGraph merges it with the existing state.

Each agent:
  1. Reads from state (immutable)
  2. Calls OllamaClient.chat() / .generate() with temperature=0.0
  3. Parses + validates the response via a Pydantic schema
  4. Writes one audit log entry (AuditDB)
  5. Returns a partial state dict
"""
