"""Add request_data and response_data columns to log_entries

Revision ID: 004
Revises: 003
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add request_data column to store full Portkey request payload
    op.add_column('log_entries', sa.Column('request_data', sa.JSON(), nullable=True))
    # Add response_data column to store full Portkey response payload
    op.add_column('log_entries', sa.Column('response_data', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('log_entries', 'response_data')
    op.drop_column('log_entries', 'request_data')
