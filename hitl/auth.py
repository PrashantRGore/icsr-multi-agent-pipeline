"""
hitl/auth.py
============
FastAPI authentication dependency for the HITL review server.

Provides equire_reviewer — a dependency that reads the X-API-Key request
header, validates it against AuthDB, and returns the authenticated ReviewerRecord.

Usage in routes:
    @router.post("/review/{review_id}/submit")
    def submit_correction(
        review_id: str,
        body:      HITLCorrectionRequest,
        reviewer:  ReviewerRecord = Depends(require_reviewer),
        ...
    ):
        # reviewer.reviewer_id is cryptographically proven — not self-reported

21 CFR Part 11 §11.50: Signed records must link to the signer's identity.
Using a key-derived reviewer_id ensures the link is tamper-proof.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from infra.auth_db import AuthDB, ReviewerRecord


def get_auth_db(request: Request) -> AuthDB:
    """Extract AuthDB from FastAPI application state."""
    auth_db = getattr(request.app.state, "auth_db", None)
    if auth_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service not initialised.",
        )
    return auth_db


def require_reviewer(
    x_api_key: str = Header(
        ...,
        description="Reviewer API key (issued by the system administrator via manage_reviewers.py)",
        alias="X-API-Key",
    ),
    auth_db: AuthDB = Depends(get_auth_db),
) -> ReviewerRecord:
    """
    FastAPI dependency: validates X-API-Key header and returns ReviewerRecord.

    Raises HTTP 401 if the key is missing, invalid, or belongs to an inactive reviewer.
    The ReviewerRecord is injected into the route handler — reviewer_id is then
    derived from the key, closing the identity-forgery vector.
    """
    reviewer = auth_db.authenticate(x_api_key)
    if reviewer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return reviewer
