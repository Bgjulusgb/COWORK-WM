"""initial baseline

Revision ID: 001_initial
Revises:
Create Date: 2026-05-31

The MVP shipped with SQLite + an in-line `_add_missing_columns()` shim.
This baseline marks "wherever the live DB currently is" so future
migrations can build on a stable starting point.

When introducing PostgreSQL: empty schema → `alembic upgrade head` will
materialise this baseline + all subsequent revisions.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Baseline: tables are created by SQLAlchemy's `Base.metadata.create_all`
    # at startup (see db.database.init_db). This revision is intentionally
    # a no-op so existing SQLite installs can be stamped with
    #   alembic stamp 001_initial
    # without rewriting their schema.
    pass


def downgrade() -> None:
    pass
