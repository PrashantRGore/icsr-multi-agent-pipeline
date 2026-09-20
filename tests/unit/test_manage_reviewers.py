"""
tests/unit/test_manage_reviewers.py
=====================================
Unit tests for scripts/manage_reviewers.py CLI.

Note: --db-path is a global flag on the root parser and must appear
BEFORE the subcommand name in argv (standard argparse behaviour).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.manage_reviewers import main, build_parser


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_reviewer():
    r = MagicMock()
    r.reviewer_id  = "QPPV-01"
    r.role         = "QPPV"
    r.created_at   = "2026-09-01T10:00:00+00:00"
    r.last_used_at = "2026-09-10T08:30:00+00:00"
    return r


@pytest.fixture
def mock_db(mock_reviewer):
    db = MagicMock()
    db.create_reviewer.return_value = "raw-api-key-xyz"
    db.list_active.return_value     = [mock_reviewer]
    db.rotate_key.return_value      = "new-api-key-abc"
    db.deactivate.return_value      = None
    return db


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:
    def test_add_subcommand_parsed(self):
        parser = build_parser()
        args   = parser.parse_args(["add", "--reviewer-id", "QPPV-01", "--role", "QPPV"])
        assert args.command     == "add"
        assert args.reviewer_id == "QPPV-01"
        assert args.role        == "QPPV"

    def test_list_subcommand_parsed(self):
        parser = build_parser()
        args   = parser.parse_args(["list"])
        assert args.command == "list"

    def test_deactivate_subcommand_parsed(self):
        parser = build_parser()
        args   = parser.parse_args(["deactivate", "--reviewer-id", "QPPV-01"])
        assert args.command     == "deactivate"
        assert args.reviewer_id == "QPPV-01"

    def test_rotate_subcommand_parsed(self):
        parser = build_parser()
        args   = parser.parse_args(["rotate", "--reviewer-id", "QPPV-01"])
        assert args.command     == "rotate"
        assert args.reviewer_id == "QPPV-01"

    def test_default_role_is_reviewer(self):
        parser = build_parser()
        args   = parser.parse_args(["add", "--reviewer-id", "MED-01"])
        assert args.role == "REVIEWER"

    def test_invalid_role_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["add", "--reviewer-id", "X", "--role", "SUPERUSER"])

    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_db_path_before_subcommand(self):
        """--db-path is a global flag — must precede the subcommand."""
        parser = build_parser()
        args   = parser.parse_args(["--db-path", "audit/auth.db", "list"])
        assert args.db_path == "audit/auth.db"

    def test_legacy_create_alias(self):
        """'create' is a backward-compatible alias for 'add'."""
        parser = build_parser()
        args   = parser.parse_args(["create", "--reviewer-id", "QPPV-01"])
        assert args.command == "create"


# ── Command handler tests ─────────────────────────────────────────────────────

class TestCmdAdd:
    def test_add_prints_api_key(self, mock_db, capsys):
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            # --db-path BEFORE the subcommand
            rc = main(["--db-path", "audit/auth.db",
                       "add", "--reviewer-id", "QPPV-01", "--role", "QPPV"])
        out = capsys.readouterr().out
        assert "raw-api-key-xyz" in out
        assert rc == 0

    def test_add_warns_save_key(self, mock_db, capsys):
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            main(["--db-path", "audit/auth.db",
                  "add", "--reviewer-id", "QPPV-01"])
        out = capsys.readouterr().out
        assert "once" in out.lower() or "save" in out.lower()

    def test_add_db_error_returns_1(self, mock_db, capsys):
        mock_db.create_reviewer.side_effect = ValueError("duplicate reviewer")
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db",
                       "add", "--reviewer-id", "QPPV-01"])
        assert rc == 1


class TestCmdList:
    def test_list_shows_reviewer(self, mock_db, capsys):
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db", "list"])
        out = capsys.readouterr().out
        assert "QPPV-01" in out
        assert rc == 0

    def test_list_empty(self, mock_db, capsys):
        mock_db.list_active.return_value = []
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db", "list"])
        out = capsys.readouterr().out
        assert "No active" in out
        assert rc == 0


class TestCmdDeactivate:
    def test_deactivate_success(self, mock_db, capsys):
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db",
                       "deactivate", "--reviewer-id", "QPPV-01"])
        assert rc == 0
        mock_db.deactivate.assert_called_once_with("QPPV-01")

    def test_deactivate_not_found_returns_1(self, mock_db, capsys):
        mock_db.list_active.return_value = []   # no active reviewers
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db",
                       "deactivate", "--reviewer-id", "GHOST-01"])
        assert rc == 1


class TestCmdRotate:
    def test_rotate_prints_new_key(self, mock_db, capsys):
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db",
                       "rotate", "--reviewer-id", "QPPV-01"])
        out = capsys.readouterr().out
        assert "new-api-key-abc" in out
        assert rc == 0

    def test_rotate_not_found_returns_1(self, mock_db, capsys):
        mock_db.list_active.return_value = []
        with patch("scripts.manage_reviewers.AuthDB", return_value=mock_db):
            rc = main(["--db-path", "audit/auth.db",
                       "rotate", "--reviewer-id", "GHOST-01"])
        assert rc == 1
