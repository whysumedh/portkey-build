"""Make log_entries.project_id nullable for workspace-level logs

Revision ID: 003
Revises: 002
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make project_id nullable to allow workspace-level logs (log pool)
    op.alter_column('log_entries', 'project_id',
                    existing_type=sa.UUID(),
                    nullable=True)


def downgrade() -> None:
    # First delete any logs without a project_id
    op.execute("DELETE FROM log_entries WHERE project_id IS NULL")
    # Then make project_id required again
    op.alter_column('log_entries', 'project_id',
                    existing_type=sa.UUID(),
                    nullable=False)
