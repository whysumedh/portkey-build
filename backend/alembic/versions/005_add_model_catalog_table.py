"""Add model_catalog table

Revision ID: 005
Revises: 004
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_catalog',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider', sa.String(100), nullable=False),
        sa.Column('model', sa.String(200), nullable=False),
        sa.Column('display_name', sa.String(200), nullable=True),
        sa.Column('input_price_per_token', sa.Float(), nullable=False, default=0.0),
        sa.Column('output_price_per_token', sa.Float(), nullable=False, default=0.0),
        sa.Column('cache_read_price_per_token', sa.Float(), nullable=True),
        sa.Column('cache_write_price_per_token', sa.Float(), nullable=True),
        sa.Column('context_window', sa.Integer(), nullable=True),
        sa.Column('max_output_tokens', sa.Integer(), nullable=True),
        sa.Column('supports_vision', sa.Boolean(), default=False),
        sa.Column('supports_function_calling', sa.Boolean(), default=False),
        sa.Column('supports_streaming', sa.Boolean(), default=True),
        sa.Column('supports_json_mode', sa.Boolean(), default=False),
        sa.Column('model_type', sa.String(50), nullable=True),
        sa.Column('model_family', sa.String(100), nullable=True),
        sa.Column('pricing_config', sa.JSON(), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_deprecated', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    
    # Create indexes
    op.create_index('ix_model_catalog_provider', 'model_catalog', ['provider'])
    op.create_index('ix_model_catalog_model', 'model_catalog', ['model'])
    op.create_index('ix_model_catalog_provider_model', 'model_catalog', ['provider', 'model'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_model_catalog_provider_model', 'model_catalog')
    op.drop_index('ix_model_catalog_model', 'model_catalog')
    op.drop_index('ix_model_catalog_provider', 'model_catalog')
    op.drop_table('model_catalog')
