"""Add tier, use_cases, quality/speed scores to model catalog

Revision ID: 006
Revises: 005
Create Date: 2026-01-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add tier classification
    op.add_column('model_catalog', sa.Column('tier', sa.String(20), nullable=True))
    
    # Add use cases (JSON array)
    op.add_column('model_catalog', sa.Column('use_cases', sa.JSON(), nullable=True))
    
    # Add quality and speed scores
    op.add_column('model_catalog', sa.Column('quality_score', sa.Integer(), nullable=True))
    op.add_column('model_catalog', sa.Column('speed_score', sa.Integer(), nullable=True))
    
    # Add recommended_for text
    op.add_column('model_catalog', sa.Column('recommended_for', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('model_catalog', 'recommended_for')
    op.drop_column('model_catalog', 'speed_score')
    op.drop_column('model_catalog', 'quality_score')
    op.drop_column('model_catalog', 'use_cases')
    op.drop_column('model_catalog', 'tier')
