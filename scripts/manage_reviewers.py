"""
scripts/manage_reviewers.py
============================
CLI for managing HITL reviewer API keys.

Usage:
  python scripts/manage_reviewers.py add --reviewer-id QPPV-01 --role QPPV
  python scripts/manage_reviewers.py list
  python scripts/manage_reviewers.py deactivate --reviewer-id QPPV-01
  python scripts/manage_reviewers.py rotate --reviewer-id QPPV-01

  # Legacy aliases (backward-compatible)
  python scripts/manage_reviewers.py create --id QPPV-01 --role QPPV

Environment:
  AUTH_DB_PATH  — SQLite auth DB path (default: audit/auth.db)

Exit codes:
  0 — success
  1 — command error (reviewer not found, duplicate, etc.)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as both `python scripts/manage_reviewers.py` and
# `python -m scripts.manage_reviewers` from project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infra.auth_db import AuthDB


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_db(args: argparse.Namespace) -> AuthDB:
    db_path = getattr(args, "db_path", None) or os.getenv("AUTH_DB_PATH", "audit/auth.db")
    return AuthDB(db_path=db_path)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace) -> int:
    """Create a new reviewer and print the raw API key (shown exactly once)."""
    db = _get_db(args)
    try:
        raw_key = db.create_reviewer(
            reviewer_id=args.reviewer_id,
            role=args.role.upper(),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Reviewer created successfully.")
    print(f"  reviewer_id : {args.reviewer_id}")
    print(f"  role        : {args.role.upper()}")
    print(f"  api_key     : {raw_key}")
    print()
    print("WARNING: Save this key — it is shown exactly once and cannot be recovered.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all active reviewers (no keys exposed)."""
    db     = _get_db(args)
    active = db.list_active()
    if not active:
        print("No active reviewers found.")
        return 0

    print(f"Active reviewers ({len(active)}):")
    print("-" * 48)
    for r in active:
        last = r.last_used_at or "never"
        print(f"  reviewer_id : {r.reviewer_id}")
        print(f"  role        : {r.role}")
        print(f"  created_at  : {r.created_at}")
        print(f"  last_used   : {last}")
        print("-" * 48)
    return 0


def cmd_deactivate(args: argparse.Namespace) -> int:
    """Soft-delete a reviewer (preserves audit trail)."""
    db     = _get_db(args)
    active = {r.reviewer_id for r in db.list_active()}
    if args.reviewer_id not in active:
        print(
            f"ERROR: No active reviewer found with id={args.reviewer_id!r}",
            file=sys.stderr,
        )
        return 1
    try:
        db.deactivate(args.reviewer_id)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Reviewer {args.reviewer_id!r} deactivated. They can no longer authenticate.")
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    """Rotate the API key for an existing active reviewer."""
    db     = _get_db(args)
    active = {r.reviewer_id for r in db.list_active()}
    if args.reviewer_id not in active:
        print(
            f"ERROR: No active reviewer found with id={args.reviewer_id!r}",
            file=sys.stderr,
        )
        return 1
    try:
        new_key = db.rotate_key(args.reviewer_id)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Key rotated for reviewer {args.reviewer_id!r}.")
    print(f"  api_key : {new_key}")
    print()
    print("WARNING: The old key is now invalid. Save this new key — shown exactly once.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser (public so tests can import it)
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_reviewers",
        description="ICSR HITL reviewer API-key lifecycle management",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        metavar="PATH",
        help="Path to auth.db (default: AUTH_DB_PATH env or audit/auth.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # add (primary)
    p_add = sub.add_parser("add", help="Create a new reviewer and print the API key")
    p_add.add_argument("--reviewer-id", required=True,
                       help="Role-based ID, e.g. QPPV-01")
    p_add.add_argument("--role", default="REVIEWER",
                       choices=["REVIEWER", "QPPV", "ADMIN"],
                       help="Reviewer role (default: REVIEWER)")

    # create — legacy alias for add (same handler, same flags)
    p_create = sub.add_parser("create", help="Alias for 'add' (legacy)")
    p_create.add_argument("--reviewer-id", "--id", dest="reviewer_id", required=True)
    p_create.add_argument("--role", default="REVIEWER",
                          choices=["REVIEWER", "QPPV", "ADMIN"])

    # list
    sub.add_parser("list", help="List all active reviewers")

    # deactivate
    p_deac = sub.add_parser("deactivate", help="Deactivate a reviewer")
    p_deac.add_argument("--reviewer-id", "--id", dest="reviewer_id", required=True)

    # rotate
    p_rot = sub.add_parser("rotate", help="Rotate the API key for a reviewer")
    p_rot.add_argument("--reviewer-id", "--id", dest="reviewer_id", required=True)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

_COMMANDS = {
    "add":        cmd_add,
    "create":     cmd_add,   # legacy alias
    "list":       cmd_list,
    "deactivate": cmd_deactivate,
    "rotate":     cmd_rotate,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
