"""add error to log inference

Revision ID: d622ff99892b
Revises: a71188c2a1ac
Create Date: 2025-08-10 08:39:36.170258

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd622ff99892b'
down_revision: Union[str, Sequence[str], None] = 'a71188c2a1ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    op.add_column('inferencelog', sa.Column('error', sa.Text(), nullable=True))

def downgrade():
    op.drop_column('inferencelog', 'error')

