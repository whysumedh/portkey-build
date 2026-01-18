"""Add LiveBench integration - capability_expectations table and benchmark columns

Revision ID: 008
Revises: 007
Create Date: 2026-01-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '008'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Get connection to check existing objects
    conn = op.get_bind()
    
    # Check if capability_expectations table already exists
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'capability_expectations')"
    ))
    table_exists = result.scalar()
    
    if not table_exists:
        # Create capability_expectations table
        op.create_table(
            'capability_expectations',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('project_id', UUID(as_uuid=True), nullable=False, unique=True),
            
            # LiveBench-aligned dimensions (scores 0-100, minimum expectations)
            sa.Column('reasoning', sa.Float(), nullable=True),
            sa.Column('coding', sa.Float(), nullable=True),
            sa.Column('agentic_coding', sa.Float(), nullable=True),
            sa.Column('mathematics', sa.Float(), nullable=True),
            sa.Column('data_analysis', sa.Float(), nullable=True),
            sa.Column('language', sa.Float(), nullable=True),
            sa.Column('instruction_following', sa.Float(), nullable=True),
            
            # Timestamps
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
            
            # Foreign key constraint
            sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        )
    
    # Helper function to check if column exists
    def column_exists(table_name, column_name):
        result = conn.execute(sa.text(
            f"SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = '{table_name}' AND column_name = '{column_name}')"
        ))
        return result.scalar()
    
    # Add LiveBench benchmark score columns to model_catalog (only if they don't exist)
    livebench_columns = [
        ('livebench_reasoning', sa.Float()),
        ('livebench_coding', sa.Float()),
        ('livebench_agentic_coding', sa.Float()),
        ('livebench_mathematics', sa.Float()),
        ('livebench_data_analysis', sa.Float()),
        ('livebench_language', sa.Float()),
        ('livebench_instruction_following', sa.Float()),
        ('livebench_global_avg', sa.Float()),
        ('livebench_last_updated', sa.DateTime(timezone=True)),
    ]
    
    for col_name, col_type in livebench_columns:
        if not column_exists('model_catalog', col_name):
            op.add_column('model_catalog', sa.Column(col_name, col_type, nullable=True))


def downgrade() -> None:
    # Remove LiveBench columns from model_catalog
    op.drop_column('model_catalog', 'livebench_last_updated')
    op.drop_column('model_catalog', 'livebench_global_avg')
    op.drop_column('model_catalog', 'livebench_instruction_following')
    op.drop_column('model_catalog', 'livebench_language')
    op.drop_column('model_catalog', 'livebench_data_analysis')
    op.drop_column('model_catalog', 'livebench_mathematics')
    op.drop_column('model_catalog', 'livebench_agentic_coding')
    op.drop_column('model_catalog', 'livebench_coding')
    op.drop_column('model_catalog', 'livebench_reasoning')
    
    # Drop capability_expectations table
    op.drop_table('capability_expectations')
