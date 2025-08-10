"""add parseDocumentPDF to tasktype enum

Revision ID: a71188c2a1ac
Revises: 4ec638b67e77
Create Date: 2025-08-10 07:39:19.133825

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a71188c2a1ac'
down_revision: Union[str, Sequence[str], None] = '4ec638b67e77'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'parseDocumentPDF'")
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
