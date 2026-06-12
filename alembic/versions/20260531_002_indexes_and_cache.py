"""indexes + translation_cache + last_crawled_at

Revision ID: 002_indexes_and_cache
Revises: 001_initial
Create Date: 2026-05-31

Adds the indexes from IMPROVE-11 and the translation_cache table from
IMPROVE-10. Safe on PostgreSQL and SQLite — `IF NOT EXISTS` clauses are
emitted where the dialect supports them.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002_indexes_and_cache"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # Add last_crawled_at column if missing.
    cols = {c["name"] for c in insp.get_columns("wm2026_matches")}
    if "last_crawled_at" not in cols:
        op.add_column("wm2026_matches", sa.Column("last_crawled_at", sa.DateTime, nullable=True))

    # IMPROVE-11: indexes (composite + single-column).
    existing = {ix["name"] for ix in insp.get_indexes("reddit_posts")}
    if "ix_reddit_posts_match_id" not in existing:
        op.create_index("ix_reddit_posts_match_id", "reddit_posts", ["match_id"])
    if "ix_reddit_posts_match_created" not in existing:
        op.create_index("ix_reddit_posts_match_created", "reddit_posts", ["match_id", "created_utc"])

    existing = {ix["name"] for ix in insp.get_indexes("sentiment_scores")}
    if "ix_sentiment_scores_match_id" not in existing:
        op.create_index("ix_sentiment_scores_match_id", "sentiment_scores", ["match_id"])

    existing = {ix["name"] for ix in insp.get_indexes("sentiment_snapshots")}
    if "ix_sentiment_snapshots_match_time" not in existing:
        op.create_index("ix_sentiment_snapshots_match_time", "sentiment_snapshots", ["match_id", "snapshot_time"])

    existing = {ix["name"] for ix in insp.get_indexes("match_predictions")}
    if "ix_match_predictions_match_time" not in existing:
        op.create_index("ix_match_predictions_match_time", "match_predictions", ["match_id", "generated_at"])

    # IMPROVE-10: translation_cache table.
    if "translation_cache" not in insp.get_table_names():
        op.create_table(
            "translation_cache",
            sa.Column("id", sa.String, primary_key=True),
            sa.Column("source_lang", sa.String(8), index=True),
            sa.Column("text_hash_prefix", sa.String(16)),
            sa.Column("translated", sa.Text),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        )


def downgrade() -> None:
    op.drop_table("translation_cache")
    op.drop_index("ix_match_predictions_match_time", table_name="match_predictions")
    op.drop_index("ix_sentiment_snapshots_match_time", table_name="sentiment_snapshots")
    op.drop_index("ix_sentiment_scores_match_id", table_name="sentiment_scores")
    op.drop_index("ix_reddit_posts_match_created", table_name="reddit_posts")
    op.drop_index("ix_reddit_posts_match_id", table_name="reddit_posts")
    op.drop_column("wm2026_matches", "last_crawled_at")
