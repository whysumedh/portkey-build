"""Add selected_log_ids and log_filter_metadata to projects

Revision ID: 002
Revises: 001
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = None  # First migration
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add selected_log_ids column (JSON array of log IDs)
    op.add_column('projects', sa.Column('selected_log_ids', sa.JSON(), nullable=True))
    # Add log_filter_metadata column (JSON object for metadata-based filtering)
    op.add_column('projects', sa.Column('log_filter_metadata', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'log_filter_metadata')
    op.drop_column('projects', 'selected_log_ids')
