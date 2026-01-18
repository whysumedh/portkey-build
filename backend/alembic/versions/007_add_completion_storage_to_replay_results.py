"""Add completion_text and original_completion to replay_results

Revision ID: 007
Revises: 006
Create Date: 2026-01-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add completion_text to store the replay response for AI judge evaluation
    op.add_column(
        'replay_results',
        sa.Column('completion_text', sa.Text(), nullable=True)
    )
    
    # Add original_completion to store the original response for comparison
    op.add_column(
        'replay_results',
        sa.Column('original_completion', sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('replay_results', 'original_completion')
    op.drop_column('replay_results', 'completion_text')
