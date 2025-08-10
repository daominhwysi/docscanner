"""add name field to task

Revision ID: 4ec638b67e77
Revises: cc1f0ebdbeac
Create Date: 2025-08-10 07:36:37.304495

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4ec638b67e77'
down_revision: Union[str, Sequence[str], None] = 'cc1f0ebdbeac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

from alembic import op

def upgrade():
    op.execute("ALTER TYPE tasktype ADD VALUE 'parseDocumentPDF'")

def downgrade():
    # PostgreSQL không dễ xóa value enum, thường phải recreate type
    pass
